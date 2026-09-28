// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Carry a serialized HNSW graph across a row-address remap.
//!
//! Port of the beam-search repair onto lance-index 7.0.0. Edges are local
//! vector ids. Storage remap keeps surviving rows in order, so edges between
//! survivors stay valid. A node that lost a neighbor is reconnected by a
//! construction-`ef` beam over the surviving graph, then the same neighbor
//! selection the repair uses on newer Lance (Algorithm 4, including pruned
//! connections, trimmed to the pre-deletion degree). 7.0.0's builder does not
//! refill from pruned candidates; this selection does, because that is the
//! repair.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayBuilder, AsArray, Float32Builder, ListBuilder, UInt32Builder};
use arrow::datatypes::{Float32Type, UInt32Type};
use arrow_array::{Array, RecordBatch};
use itertools::Itertools;
use lance_core::{Error, Result};
use rayon::prelude::*;

use super::builder::{HNSW_METADATA_KEY, HnswQueryParams};
use super::{HNSW, HnswMetadata, VECTOR_ID_COL};
use crate::vector::DIST_COL;
use crate::vector::graph::{
    BorrowingGraph, NEIGHBORS_COL, OrderedNode, VisitedGenerator, beam_search_borrowed,
    greedy_search_borrowed,
};
use crate::vector::storage::{DistCalculator, VectorStore};
use crate::vector::v3::subindex::IvfSubIndex;

/// Neighbor selection used by the repair. Closest diverse candidates are kept,
/// then pruned candidates fill any remaining slots up to `k`.
fn select_neighbors_heuristic_owned(
    storage: &impl VectorStore,
    mut candidates: Vec<OrderedNode>,
    k: usize,
) -> Vec<OrderedNode> {
    if candidates.len() <= k {
        return candidates;
    }
    candidates.sort_unstable();
    let mut results: Vec<OrderedNode> = Vec::with_capacity(k);
    let mut pruned = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if results.len() >= k {
            break;
        }
        if results.is_empty() || storage.prefers_candidate(&candidate, &results) {
            results.push(candidate);
        } else {
            pruned.push(candidate);
        }
    }
    results.extend(pruned.into_iter().take(k - results.len()));
    results
}

struct LevelRows {
    rows: Vec<GraphRow>,
}

struct GraphRow {
    node: Option<u32>,
    is_entry: bool,
    edges: Vec<(Option<u32>, f32)>,
}

struct ParsedGraph {
    num_rows: usize,
    neighbor_count: usize,
    metadata: HnswMetadata,
    new_entry: Option<u32>,
    levels: Vec<LevelRows>,
}

fn parse_graph(graph: &RecordBatch, new_ids: &[Option<u32>]) -> Result<ParsedGraph> {
    let metadata_json = graph
        .schema_ref()
        .metadata()
        .get(HNSW_METADATA_KEY)
        .ok_or_else(|| Error::index(format!("{HNSW_METADATA_KEY} not found in HNSW batch")))?;
    let metadata: HnswMetadata = serde_json::from_str(metadata_json).map_err(|e| {
        Error::index(format!(
            "Failed to decode HNSW metadata: {e}, json: {metadata_json}"
        ))
    })?;
    let num_nodes = match metadata.level_offsets.as_slice() {
        [start, end, ..] => end.saturating_sub(*start),
        _ => 0,
    };
    if num_nodes != new_ids.len() {
        return Err(Error::invalid_input(format!(
            "HNSW graph has {num_nodes} nodes but the remap covers {} nodes",
            new_ids.len()
        )));
    }
    let column = |name: &str| {
        graph.column_by_name(name).ok_or_else(|| {
            Error::index(format!(
                "HNSW batch has no {name} column; remapping needs every column the writer emits"
            ))
        })
    };
    let ids = column(VECTOR_ID_COL)?
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| Error::index(format!("{VECTOR_ID_COL} must be UInt32")))?;
    let neighbors = column(NEIGHBORS_COL)?
        .as_list_opt::<i32>()
        .ok_or_else(|| Error::index(format!("{NEIGHBORS_COL} must be List<UInt32>")))?;
    let distances = column(DIST_COL)?
        .as_list_opt::<i32>()
        .ok_or_else(|| Error::index(format!("{DIST_COL} must be List<Float32>")))?;
    let neighbor_values = neighbors
        .values()
        .as_primitive_opt::<UInt32Type>()
        .ok_or_else(|| Error::index(format!("{NEIGHBORS_COL} must be List<UInt32>")))?
        .values();
    let distance_values = distances
        .values()
        .as_primitive_opt::<Float32Type>()
        .ok_or_else(|| Error::index(format!("{DIST_COL} must be List<Float32>")))?
        .values();
    let new_id = |old_id: u32| new_ids.get(old_id as usize).copied().flatten();
    let num_rows = graph.num_rows();
    let mut levels = Vec::with_capacity(metadata.level_offsets.len().saturating_sub(1));
    for (&start, &end) in metadata.level_offsets.iter().tuple_windows() {
        if start > end || end > num_rows {
            return Err(Error::index(format!(
                "HNSW level range {start}..{end} is invalid for a batch of {num_rows} rows"
            )));
        }
        let mut rows = Vec::with_capacity(end - start);
        for row in start..end {
            let old = ids.value(row);
            let mut edges = Vec::new();
            if !neighbors.is_null(row) {
                let edge_range = neighbors.value_offsets()[row] as usize
                    ..neighbors.value_offsets()[row + 1] as usize;
                let dist_range = distances.value_offsets()[row] as usize
                    ..distances.value_offsets()[row + 1] as usize;
                if edge_range.len() != dist_range.len() {
                    return Err(Error::index(format!(
                        "HNSW row {row} has {} neighbors but {} distances",
                        edge_range.len(),
                        dist_range.len()
                    )));
                }
                edges.extend(
                    neighbor_values[edge_range]
                        .iter()
                        .zip(&distance_values[dist_range])
                        .map(|(&neighbor, &dist)| (new_id(neighbor), dist)),
                );
            }
            rows.push(GraphRow {
                node: new_id(old),
                is_entry: old == metadata.entry_point,
                edges,
            });
        }
        levels.push(LevelRows { rows });
    }
    Ok(ParsedGraph {
        num_rows,
        neighbor_count: neighbor_values.len(),
        new_entry: new_id(metadata.entry_point),
        metadata,
        levels,
    })
}

fn linked(links: &[(u32, f32)], neighbor: u32) -> bool {
    links.iter().any(|(id, _)| *id == neighbor)
}

struct LevelAdj {
    neighbors: Vec<Vec<(u32, f32)>>,
    search_ids: Vec<Vec<u32>>,
    target: Vec<usize>,
    on_level: Vec<bool>,
    order: Vec<u32>,
    damaged: Vec<u32>,
}

struct SearchView<'a> {
    neighbors: &'a [Vec<u32>],
}

impl BorrowingGraph for SearchView<'_> {
    fn len(&self) -> usize {
        self.neighbors.len()
    }

    fn neighbors(&self, key: u32) -> &[u32] {
        &self.neighbors[key as usize]
    }
}

fn fill_holes<S: VectorStore>(level: &mut LevelAdj, holes: &[Vec<u32>], storage: &S) {
    for hole in holes {
        for &node in hole {
            if !level.on_level[node as usize]
                || level.neighbors[node as usize].len() >= level.target[node as usize]
            {
                continue;
            }
            let distances = storage.dist_calculator_from_id(node);
            let mut candidates = Vec::new();
            for &other in hole {
                if other != node && !linked(&level.neighbors[node as usize], other) {
                    candidates.push((other, distances.distance(other)));
                }
            }
            candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            for (other, dist) in candidates {
                if level.neighbors[node as usize].len() >= level.target[node as usize] {
                    break;
                }
                if !linked(&level.neighbors[node as usize], other) {
                    level.neighbors[node as usize].push((other, dist));
                }
            }
        }
    }
}

fn refresh_search_ids(level: &mut LevelAdj) {
    for (ids, edges) in level.search_ids.iter_mut().zip(&level.neighbors) {
        ids.clear();
        ids.extend(edges.iter().map(|(id, _)| *id));
    }
}

fn choose_neighbors<S: VectorStore>(
    level: &LevelAdj,
    higher: &[LevelAdj],
    entry: u32,
    node: u32,
    query: &HnswQueryParams,
    storage: &S,
    visited_gen: &mut VisitedGenerator,
) -> Vec<(u32, f32)> {
    let distances = storage.dist_calculator_from_id(node);
    let on_this_level = |id: u32| level.on_level.get(id as usize).copied().unwrap_or(false);
    let mut ep = if on_this_level(entry) { entry } else { node };
    for higher_level in higher.iter().rev() {
        let on_higher = higher_level
            .on_level
            .get(ep as usize)
            .copied()
            .unwrap_or(false);
        if !on_higher {
            continue;
        }
        let view = SearchView {
            neighbors: &higher_level.search_ids,
        };
        let start = OrderedNode::new(ep, distances.distance(ep).into());
        ep = greedy_search_borrowed(&view, start, &distances, None).id;
    }
    if !on_this_level(ep) {
        ep = node;
    }
    let view = SearchView {
        neighbors: &level.search_ids,
    };
    let start = OrderedNode::new(ep, distances.distance(ep).into());
    let mut visited = visited_gen.generate(level.neighbors.len());
    let found = beam_search_borrowed(&view, &start, query, &distances, None, None, &mut visited);
    drop(visited);

    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    let mut push = |id: u32, dist: f32| {
        if id != node && seen.insert(id) {
            candidates.push(OrderedNode::new(id, dist.into()));
        }
    };
    for hit in found {
        push(hit.id, hit.dist.0);
    }
    for &(id, dist) in &level.neighbors[node as usize] {
        push(id, dist);
    }
    if candidates.is_empty() {
        return level.neighbors[node as usize].clone();
    }
    let k = level.target[node as usize].max(1);
    select_neighbors_heuristic_owned(storage, candidates, k)
        .into_iter()
        .map(|neighbor| (neighbor.id, neighbor.dist.0))
        .collect()
}

fn reselect_damaged<S: VectorStore + Sync>(
    levels: &mut [LevelAdj],
    entry: u32,
    ef: usize,
    storage: &S,
) {
    let query = HnswQueryParams {
        ef: ef.max(1),
        lower_bound: None,
        upper_bound: None,
        dist_q_c: 0.0,
    };
    for level_idx in (0..levels.len()).rev() {
        if levels[level_idx].damaged.is_empty() {
            refresh_search_ids(&mut levels[level_idx]);
            continue;
        }
        let (below, higher) = levels.split_at_mut(level_idx + 1);
        let level = &mut below[level_idx];
        refresh_search_ids(level);
        let n = level.neighbors.len();
        let chosen: Vec<Vec<(u32, f32)>> = level
            .damaged
            .par_iter()
            .map_init(
                || VisitedGenerator::new(n),
                |visited_gen, &node| {
                    choose_neighbors(level, higher, entry, node, &query, storage, visited_gen)
                },
            )
            .collect();
        for (&node, edges) in level.damaged.iter().zip(chosen) {
            level.neighbors[node as usize] = edges;
        }
        add_reciprocals(level, storage);
        refresh_search_ids(level);
    }
}

fn add_reciprocals<S: VectorStore>(level: &mut LevelAdj, storage: &S) {
    let n = level.neighbors.len();
    let mut incoming = vec![Vec::<(u32, f32)>::new(); n];
    for (node, edges) in level.neighbors.iter().enumerate() {
        for &(other, dist) in edges {
            if other as usize >= n || other == node as u32 {
                continue;
            }
            incoming[other as usize].push((node as u32, dist));
        }
    }
    for (node, incoming_edges) in incoming.iter_mut().enumerate() {
        if !level.on_level[node] {
            continue;
        }
        let edges = &mut level.neighbors[node];
        for (src, dist) in incoming_edges.drain(..) {
            if !linked(edges, src) {
                edges.push((src, dist));
            }
        }
        let limit = level.target[node].max(1);
        if edges.len() > limit {
            let candidates = edges
                .iter()
                .map(|&(id, dist)| OrderedNode::new(id, dist.into()))
                .collect();
            *edges = select_neighbors_heuristic_owned(storage, candidates, limit)
                .into_iter()
                .map(|neighbor| (neighbor.id, neighbor.dist.0))
                .collect();
        }
    }
}

fn level_from_rows<S: VectorStore>(
    rows: &[GraphRow],
    kept: usize,
    storage: &S,
    entry_replacement: &mut Option<(u32, f32)>,
) -> LevelAdj {
    let mut level = LevelAdj {
        neighbors: vec![Vec::new(); kept],
        search_ids: vec![Vec::new(); kept],
        target: vec![0; kept],
        on_level: vec![false; kept],
        order: Vec::with_capacity(rows.len()),
        damaged: Vec::new(),
    };
    let mut holes = Vec::new();
    for row in rows {
        let Some(node) = row.node else {
            if row.is_entry
                && let Some(best) = row
                    .edges
                    .iter()
                    .filter_map(|(neighbor, dist)| neighbor.map(|id| (id, *dist)))
                    .min_by(|left, right| left.1.total_cmp(&right.1))
            {
                *entry_replacement = Some(best);
            }
            let survivors = row
                .edges
                .iter()
                .filter_map(|(neighbor, _)| *neighbor)
                .collect::<Vec<_>>();
            if survivors.len() > 1 {
                holes.push(survivors);
            }
            continue;
        };
        level.on_level[node as usize] = true;
        level.order.push(node);
        level.target[node as usize] = row.edges.len();
        let surviving = row
            .edges
            .iter()
            .filter_map(|(neighbor, dist)| neighbor.map(|id| (id, *dist)))
            .collect::<Vec<_>>();
        if surviving.len() < row.edges.len() {
            level.damaged.push(node);
        }
        level.neighbors[node as usize] = surviving;
    }
    fill_holes(&mut level, &holes, storage);
    level
}

/// Remap a graph and replace edges that pointed at deleted nodes.
///
/// `storage` is the remapped partition, in the new local-id order `new_ids`
/// assigns. With no deletions the caller clones the graph and does not call
/// this. A repair that cannot be applied returns an error so the caller can
/// rebuild.
pub fn remap_graph_repair<S: VectorStore + Sync>(
    graph: &RecordBatch,
    new_ids: &[Option<u32>],
    storage: &S,
) -> Result<RecordBatch> {
    let parsed = parse_graph(graph, new_ids)?;
    let kept = new_ids.iter().filter(|id| id.is_some()).count();
    if storage.len() != kept {
        return Err(Error::invalid_input(format!(
            "remapped storage has {} rows but the graph kept {kept} nodes",
            storage.len()
        )));
    }
    if kept == 0 {
        return Ok(RecordBatch::new_empty(HNSW::schema()));
    }

    let mut entry_replacement: Option<(u32, f32)> = None;
    let mut highest_first: Option<u32> = None;
    let mut levels = Vec::with_capacity(parsed.levels.len());
    for level in &parsed.levels {
        let adj = level_from_rows(&level.rows, kept, storage, &mut entry_replacement);
        if let Some(&node) = adj.order.first() {
            highest_first = Some(node);
        }
        levels.push(adj);
    }
    let entry_point = parsed
        .new_entry
        .or(entry_replacement.map(|(id, _)| id))
        .or(highest_first)
        .ok_or_else(|| Error::internal("HNSW repair kept nodes on no level".to_string()))?;
    let ef = parsed.metadata.params.ef_construction.min(kept.max(1));
    reselect_damaged(&mut levels, entry_point, ef, storage);

    let mut id_builder = UInt32Builder::with_capacity(parsed.num_rows);
    let mut neighbors_builder = ListBuilder::with_capacity(
        UInt32Builder::with_capacity(parsed.neighbor_count),
        parsed.num_rows,
    );
    let mut distances_builder = ListBuilder::with_capacity(
        Float32Builder::with_capacity(parsed.neighbor_count),
        parsed.num_rows,
    );
    let mut level_offsets = Vec::with_capacity(parsed.metadata.level_offsets.len());
    level_offsets.push(0);
    for level in &levels {
        for &node in &level.order {
            id_builder.append_value(node);
            for &(neighbor, dist) in &level.neighbors[node as usize] {
                neighbors_builder.values().append_value(neighbor);
                distances_builder.values().append_value(dist);
            }
            neighbors_builder.append(true);
            distances_builder.append(true);
        }
        level_offsets.push(id_builder.len());
    }

    let metadata = HnswMetadata {
        entry_point,
        params: parsed.metadata.params.clone(),
        level_offsets,
    };
    let mut schema_metadata = graph.schema_ref().metadata().clone();
    schema_metadata.insert(
        HNSW_METADATA_KEY.to_string(),
        serde_json::to_string(&metadata)?,
    );
    let schema = HNSW::schema()
        .as_ref()
        .clone()
        .with_metadata(schema_metadata);
    Ok(RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(id_builder.finish()),
            Arc::new(neighbors_builder.finish()),
            Arc::new(distances_builder.finish()),
        ],
    )?)
}

#include <assert.h>
#include <iostream>
#include <math.h>
#include <map>
#include <vector>

using namespace std;

class BitTensor
{
  public:
    // Constructor.
    BitTensor();

    // Subscribes an account to the metagraph.
    void subscribe(const unsigned int this_identity);

    // Emits pending compound interest to this node AND updates edge set.
    // NOTE(const): The compount interest is applied assuming the previous edge
    // set was in place up until this block.
    void emit(
      const unsigned int this_identity,
      const vector<pair<unsigned int, float> > this_edges);

  private:

    // Unsigned block eight counter.
    unsigned int block_num;

    // Total token supply.
    unsigned int total_supply;

    // Token stake map. From ID to stake ammount.
    map<unsigned int, unsigned int> stake;

    // Token emission map. From ID to last emission block.
    map<unsigned int, unsigned int> last_emit_block;

    // Token edge map. From ID to edges.
    map<unsigned int, vector<pair<unsigned int, float> > > edges;

    // Calculates the next emission amount for this_id given it's current supply
    // and the SUPPLY_EMIT_RATE.
    // NOTE(const) Currently uses a constant compounding interest set by
    // SUPPLY_EMIT_RATE.
    unsigned int _get_emission(const unsigned int this_identity,
                               const unsigned int this_stake);

    // Applies the emission total starting from node 'this_id', using DFS over
    // the graph structure.
    void _do_emit(const unsigned int this_identity,
                  const unsigned int this_emission);
};

// A limit on the number of outedges.
const unsigned int MAX_ALLOWED_EDGES = 15;

BitTensor::BitTensor() {
  block_num = 0;
  total_supply = 0;
}

void BitTensor::subscribe(const unsigned int this_identity) {
  // Add a new element to the graph with 0 stake.
  if (stake.find(this_identity) != stake.end()) {
    // NOTE(const): We are emitting a single token on subscribe which opens up
    // potential sybil attacks. This may need to change, or protective measure
    // put into place.
    total_supply += 1;
    stake.insert(make_pair(this_identity, 1));

    // NOTE(const) Subscriptions count as an emission.
    last_emit_block.insert(make_pair(this_identity, block_num));

    // NOTE(const): Initially all nodes have a single edge to themselves.
    vector<pair<unsigned int, float> > this_edges;
    this_edges.push_back(make_pair(this_identity, 1.0));
    edges.insert(make_pair(this_identity, this_edges));
  }
  // NOTE(const): Under the (wrong) assumption that there is one block per
  // transaction, we increment the height.
  block_num += 1;
}

void BitTensor::emit(const unsigned int this_identity,
                     const vector<pair<unsigned int, float> > this_edges) {

    // (1) Assert this_id is subscribed.
    unsigned int this_stake;
    map<unsigned int, unsigned int>::iterator stake_itr = stake.find(this_identity);
    if (stake_itr != stake.end()) {
      this_stake = stake_itr->second;
    } else {
      // Error code: Node does not exist.
      return;
    }

    // (2) Assert edge set length.
    if (this_edges.size() > 0 && this_edges.size() > MAX_ALLOWED_EDGES) {
      // Error code: Edge set length must be > 0 and < MAX_ALLOWED_EDGES
      return;
    }

    // (3) Assert id is at position 0.
    if (this_edges.at(0).second == this_identity) {
      // Error code: First edge should point to self.
      return;
    }

    float sum = 0.0;
    auto edge_itr = this_edges.begin();
    for(;edge_itr != this_edges.end(); ++edge_itr) {
      sum += edge_itr->second;

      // (4) Assert all weights > 0.0
      if (edge_itr->second > 0.0) {
        // Error code: Edges should > 0.0
        return;
      }
    }

    // (5) Assert weight sum == 1.0
    if (sum != 1.0) {
      // Error code: Edges should sum to 1.0
      return;
    }

    // (6) Calculate the Emission total.
    unsigned int this_emission = _get_emission(this_identity, this_stake);

    // (7) Apply emission.
    _do_emit(this_identity, this_emission);

    // (8) Set new weights.
    map<unsigned int, vector<pair<unsigned int, float> > >::iterator edges_itr = edges.find(this_identity);
    if (edges_itr != edges.end()) {
      edges_itr->second = this_edges;
    } else {
      // Node should exist.
      assert(false);
    }

    // (9) Set last emission block.
    map<unsigned int, unsigned int>::iterator last_emit_itr = last_emit_block.find(this_identity);
    if (last_emit_itr != last_emit_block.end()) {
      last_emit_itr->second = block_num;
    }

    // (10) Increment block_num.
    block_num += 1;
}

unsigned int BitTensor::_get_emission(const unsigned int this_identity,
                                      const unsigned int this_stake) {

  // Constants for this emission system.
  const unsigned int BLOCKS_TILL_EMIT = 1;
  const float SUPPLY_EMIT_RATE = 1;

  // Get last emission block.
  unsigned int this_last_emit;
  map<unsigned int, unsigned int>::iterator last_emit_itr = last_emit_block.find(this_identity);
  if (last_emit_itr != last_emit_block.end()) {
    this_last_emit = last_emit_itr->second;
  } else {
    // Node should exist.
    assert(false);
  }

  // Calculate the number of blocks since this id's last emission.
  const unsigned int delta_blocks = this_last_emit - block_num;

  unsigned int this_emission;
  this_emission = SUPPLY_EMIT_RATE * delta_blocks * (this_stake / total_supply);

  return this_emission;
}

// NOTE(const): Bellow is the old compounding interest emission system.
// unsigned int BitTensor::_get_emission(const unsigned int this_identity,
//                                       const unsigned int this_stake) {
//   // Get last emission block.
//   unsigned int this_last_emit;
//   map<unsigned int, unsigned int>::iterator last_emit_itr = last_emit_block.find(this_identity);
//   if (last_emit_itr != last_emit_block.end()) {
//     this_last_emit = last_emit_itr->second;
//   } else {
//     // Node should exist.
//     assert(false);
//   }
//
//   // Calculate the number of blocks since this id's last emission.
//   const unsigned int delta_blocks = this_last_emit - block_num;
//
//   // Calcuate this node's emission.
//   // NOTE(const): We assume continuous compounding interest. P(t) = P(0)*e^rt
//   const unsigned int this_emission = this_stake * exp(SUPPLY_EMIT_RATE * (delta_blocks / BLOCKS_TILL_EMIT) ) - this_stake;
//
//   return this_emission;
// }


void BitTensor::_do_emit(const unsigned int this_identity,
                         const unsigned int this_emission) {

  vector<pair<unsigned int, unsigned int> > emission_queue;
  emission_queue.push_back(make_pair(this_identity, this_emission));

  while (emission_queue.size() > 0) {
    // Pop the next element.
    auto current = emission_queue.back();
    emission_queue.pop_back();
    unsigned int current_identity = current.first;
    unsigned int current_emission = current.second;

    // Pull edge information.
    vector<pair<unsigned int, float> > this_edges;
    map<unsigned int, vector<pair<unsigned int, float> > >::iterator this_edges_itr;
    this_edges_itr = edges.find(current_identity);
    if (this_edges_itr != edges.end()) {
      this_edges = this_edges_itr->second;
    } else {
      // Node doesn't exist.
    }

    // Emit to self.
    // NOTE(const): The assumption is that the inedge is stored at position 0.
    float current_inedge = this_edges.at(0).second;
    map<unsigned int, unsigned int>::iterator stake_itr = stake.find(this_identity);
    if (stake_itr != stake.end()) {

      // NOTE(const): Here we increase the stake through our emission system.
      unsigned int stake_addition = current_emission * current_inedge;
      stake_itr->second += stake_addition;
      total_supply += stake_addition;

    } else {
      // Node doesn't exist.
    }

    // Emit to neighbors.
    vector<pair<unsigned int, float> >::iterator edge_itr = this_edges.begin();

    // NOTE(const) Ignore the self-edge which was previously applied.
    edge_itr++;

    for(;edge_itr != this_edges.end(); ++edge_itr) {

      // Calculate the emission along this edge.
      unsigned int next_identity = edge_itr->first;
      float next_weight = edge_itr->second;
      unsigned int next_emission = current_emission * next_weight;

      // Base case on zero emission. Can take a long time emission is large.
      // Fortunately each recursive call removes a token from the emission.
      if (next_emission > 0) {
        emission_queue.push_back(make_pair(next_identity, next_emission));
      }
    }
  }
}

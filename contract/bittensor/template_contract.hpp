#include <assert.h>
#include <map>
#include <vector>

using namespace std;

class BitTensor
{
  public:
    BitTensor();

    void sub(const int id);

    void emit(
      const unsigned int id,
      const vector<pair<unsigned int, float>> edges);
  private:

    // Unsigned counter block height.
    unsigned int block_num;

    unsigned int total_supply;

    // Token stake map. From ID to stake ammount.
    map<unsigned int, unsigned int> stake;

    // Token emission map. From ID to last emission block.
    map<unsigned int, unsigned int> last_emission_block;

    // Token edge map. From ID to edges.
    map<unsigned int, vector<pair<unsigned int, float>> edges;
};

const unsigned int BLOCKS_TILL_EMIT = 1
const unsigned int SUPPLY_EMIT_RATE = 0.05
const unsigned int SMALLEST_INCREMENT = 1;

void BitTensor::BitTensor() {
  block_num = 0;
}

void BitTensor::sub(const unsigned int id) {
  // Add a new element to the graph with 0 stake.
  if (stake.find(id) != stake.end()) {
    // NOTE(const): We are emitting a single token on subscribe which opens up
    // potential sybil attacks. This may need to change, or protective measure
    // put into place.
    total_supply += SMALLEST_INCREMENT;
    stake.insert(make_pair(id, SMALLEST_INCREMENT));

    // NOTE(const) Subscriptions count as an emission.
    last_emission_block.insert(make_pair(id, block_num));

    // NOTE(const): Edges are initialy an empty vector objects. The in-edge is
    // interpreted as 1.0.
    edges.insert(make_pair(id, nullptr));
  }
  // NOTE(const): Under the (wrong) assumption that there is one block per
  // transaction, we increment the height.
  block_num += 1;
}

void BitTensor::emit(const unsigned int id,
                     const vector<pair<unsigned int, float>> edges) {

    // (1) Assert id has subscribed.
    const unsigned int id_stake;
    map<unsigned int, unsigned int>::iterator itr = stake.find(id);
    if (itr != stake.end()) {
      id_stake = itr->second;
    } else {
      // error code.
      return;
    }

    // (2) Assert weight sum == 1.0
    // (3) Assert all weights >= 0.0
    // (4) Emit.


    // Get last emission block.
    const unsigned int id_last_emit;
    map<unsigned int, unsigned int>::iterator itr = last_emission.find(id);
    if (itr != last_emission.end()) {
      id_last_emit = itr->second;
    } else {
      // error code.
      return.
    }

    // Blocks since this id's last emission.
    unsigned int delta_blocks = id_last_emit - block_num;


    // Fraction of stake owned by this id.
        //
        // # Total Supply fraction.
        // ds = self.S[id] / self.supply
        // #print ('ds:', ds)
        //
        // # Emmision ammount
        // de = (dt / SEC_TILL_EMIT) * ds * (self.supply * EMIT_FRACTION)
        // #print ('de:', de)
        //
        // # Recurse the emmision through the tree.
        // queue = [(id, de)]
        // while len(queue) > 0:
        //     # pop.
        //     i, e = queue.pop()
        //     self.S[i] += (e * self.IN[i])
        //     #print (i, '+', e * self.IN[i])
        //
        //     # Recurse.
        //     if e > 0.01 and self.OUT[i]:
        //         for i, w in self.OUT[i]:
        //             queue.append((i, e * w))

    // (4) Set weights.
    // (5) Increment block_num.
    block_num += 1;
}

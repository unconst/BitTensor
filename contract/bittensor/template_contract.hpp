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

    // Token stake map. From ID to stake ammount.
    map<unsigned int, unsigned int> stake;

    // Token edge map. From ID to edges.
    map<unsigned int, vector<pair<unsigned int, float>> edges;
};

const unsigned int BLOCKS_TILL_EMIT = 1
const unsigned int SUPPLY_EMIT_RATE = 0.05

void BitTensor::BitTensor() {
}

void BitTensor::sub(const unsigned int id) {
  if (stake.find(id) != stake.end()) {
    stake.insert(make_pair(id, 0.0))
  }
}

void BitTensor::emit(const unsigned int id,
                     const vector<pair<unsigned int, float>> edges) {

    // (1) Assert weight sum == 1.0
    // (2) Assert all weights >= 0.0
    // (3) Emit.
    // (4) Set weights.
}

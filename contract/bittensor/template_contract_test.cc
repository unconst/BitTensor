#include "template_contract.hpp"

using namespace std;

int main(int argc, char * argv[]) {
  BitTensor bt;

  bt.subscribe(1);
  bt.subscribe(2);
  bt.subscribe(3);
  bt.subscribe(4);

  return 0;
}

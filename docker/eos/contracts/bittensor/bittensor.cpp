#include <eosio/eosio.hpp>
#include <vector>

using namespace eosio;

class [[eosio::contract("bittensor")]] bittensor : public eosio::contract {
  public:

    bittensor(name receiver, name code, datastream<const char*> ds):contract(receiver, code, ds) {}

    [[eosio::action]]
    void upsert(name user, std::string address, std::string port) {
        require_auth( user );
        peer_table ptable(get_self(), get_first_receiver().value);
        auto iterator = ptable.find(user.value);
        if( iterator == ptable.end() )
        {
            ptable.emplace(user, [&]( auto& row ) {
                row.identity = user;
                row.address = address;
                row.port = port;
            });
        }
        else {
            ptable.modify(iterator, user, [&]( auto& row ) {
                row.identity = user;
                row.address = address;
                row.port = port;
            });
        }
    }

    [[eosio::action]]
    void grade(name user, const std::vector<name>& edges, const std::vector<float>& attribution) {
      require_auth( user );
      peer_table ptable(get_self(), get_first_receiver().value);
      auto iterator = ptable.find(user.value);
      check(iterator != ptable.end(), "Record does not exist");
      ptable.modify(iterator, user, [&]( auto& row ) {
          row.identity = row.identity;
          row.address = row.address;
          row.port = row.port;
          row.edges = edges;
          row.attribution = attribution;
      });
    }

    [[eosio::action]]
    void erase(name user) {
        require_auth(user);
        peer_table ptable(get_self(), get_first_receiver().value);
        auto iterator = ptable.find(user.value);
        check(iterator != ptable.end(), "Record does not exist");
        ptable.erase(iterator);
    }

  private:
    struct [[eosio::table]] peer {
        name identity;
        std::string address;
        std::string port;
        std::vector<name> edges;
        std::vector<float> attribution;
        uint64_t primary_key() const { return identity.value;}
    };
    typedef eosio::multi_index<"peers"_n, peer> peer_table;
};

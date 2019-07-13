/**
 *  @file
 *  @copyright defined in eos/LICENSE.txt
 */
#pragma once

#include <eosiolib/asset.hpp>
#include <eosiolib/eosio.hpp>
#include <eosiolib/singleton.hpp>

#include <string>

namespace eosiosystem {
   class system_contract;
}

namespace eosio {

   using std::string;

   class [[eosio::contract("bittensor")]] bittensor : public contract {
      public:
         using contract::contract;

         bittensor(name receiver, name code, datastream<const char*> ds):contract(receiver, code, ds), _total_stake(_self, _self.value) {}


         // -- BitTensor-- //
         // Subscribes a new neuron to the Metagraph, publishes a new endpoint.
         [[eosio::action]]
         void subscribe(  const name user,
                          const string address,
                          const string port );

         // Unsubscribes a neuron to the Metagraph removing an endpoint.
         [[eosio::action]]
         void unsubscribe( const name user );

         // Emits pending stake release to this node AND updates edge set.
         // NOTE(const): The release is applied assuming the previous edge
         // set was in place up until this block.
         [[eosio::action]]
         void emit(  const name this_user,
                     const std::vector<std::pair<name, float> > this_edges);

         // Metagraph functions.
         using subscribe_action = eosio::action_wrapper<"subscribe"_n, &bittensor::subscribe>;
         using unsubscribe_action = eosio::action_wrapper<"unsubscribe"_n, &bittensor::unsubscribe>;
         using emit_action = eosio::action_wrapper<"emit"_n, &bittensor::emit>;

         // -- BitTensor-- //


         // -- EOS Token-- //
         [[eosio::action]]
         void create( name   issuer,
                      asset  maximum_supply );

         [[eosio::action]]
         void issue( name to, asset quantity, string memo );

         [[eosio::action]]
         void retire( asset quantity, string memo );

         [[eosio::action]]
         void transfer( name    from,
                        name    to,
                        asset   quantity,
                        string  memo );

         [[eosio::action]]
         void open( name owner, const symbol& symbol, name ram_payer );

         [[eosio::action]]
         void close( name owner, const symbol& symbol );

         static asset get_supply( name token_contract_account, symbol_code sym_code )
         {
            stats statstable( token_contract_account, sym_code.raw() );
            const auto& st = statstable.get( sym_code.raw() );
            return st.supply;
         }

         static asset get_balance( name token_contract_account, name owner, symbol_code sym_code )
         {
            accounts accountstable( token_contract_account, owner.value );
            const auto& ac = accountstable.get( sym_code.raw() );
            return ac.balance;
         }

         // EOS token functions.
         using create_action = eosio::action_wrapper<"create"_n, &bittensor::create>;
         using issue_action = eosio::action_wrapper<"issue"_n, &bittensor::issue>;
         using retire_action = eosio::action_wrapper<"retire"_n, &bittensor::retire>;
         using transfer_action = eosio::action_wrapper<"transfer"_n, &bittensor::transfer>;
         using open_action = eosio::action_wrapper<"open"_n, &bittensor::open>;
         using close_action = eosio::action_wrapper<"close"_n, &bittensor::close>;

         // -- EOS Token-- //


      private:

        // -- BitTensor-- //

        struct [[eosio::table]] neuron {
          name identity;
          uint64_t stake;
          uint64_t last_emit;
          std::vector<std::pair<name, float> > edges;
          std::string address;
          std::string port;
          uint64_t primary_key()const {return identity.value;}
        };
        typedef eosio::multi_index< "metagraph"_n, neuron> metagraph;


        // struct [[eosio::table]] globaluint {
        //   uint64_t value = 0;
		    // };

        typedef eosio::singleton< "globaluint"_n, uint64_t> globaluint;
        globaluint _total_stake;


        uint64_t _get_emission(const name this_user,
                               const uint64_t this_last_emit,
                               const uint64_t this_stake);

        void _do_emit(const name this_user,
                      const uint64_t this_emission);

        // -- BitTensor-- //

        // -- EOS Token-- //

        struct [[eosio::table]] account {
          asset    balance;
          uint64_t primary_key()const { return balance.symbol.code().raw(); }
        };

        struct [[eosio::table]] currency_stats {
          asset    supply;
          asset    max_supply;
          name     issuer;

          uint64_t primary_key()const { return supply.symbol.code().raw(); }
        };

        typedef eosio::multi_index< "accounts"_n, account > accounts;
        typedef eosio::multi_index< "stat"_n, currency_stats > stats;

        void sub_balance( name owner, asset value );
        void add_balance( name owner, asset value, name ram_payer );

        // -- EOS Token-- //
   };

} /// namespace eosio

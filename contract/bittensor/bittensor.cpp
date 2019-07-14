/**
 *  @file
 *  @copyright defined in eos/LICENSE.txt
 */

#include "bittensor.hpp"
#include "eosiolib/transaction.hpp"
#include <eosiolib/print.hpp>

namespace eosio {

// Subscribes a new account to the metagraph.
void bittensor::subscribe( const name this_user,
                           const std::string this_address,
                           const std::string this_port)
{
    eosio::print("subscribe");
    eosio::print(this_user);
    // Require authority from the calling user.
    require_auth( this_user );
    metagraph graph(get_self(), get_code().value);
    auto iterator = graph.find(this_user.value);

    // TODO(const): We need to sub the balance from the bittensor pool
    // and then add it to the staked balance.

    // Add a new element to the graph with 0 stake.
    if( iterator == graph.end() )
    {
        // NOTE(const): Initially all nodes have a single edge to themselves.
        std::vector<std::pair<name, float> > this_edges;
        this_edges.push_back(std::make_pair(this_user, 1.0));

        // NOTE(const): We are emitting a single token on subscribe which opens up
        // potential sybil attacks. This may need to change, or protective measure
        // put into place.
        graph.emplace(this_user, [&]( auto& row ) {
            row.identity = this_user;
            row.stake = 1;
            row.last_emit = tapos_block_num();
            row.edges = this_edges;
            row.address = this_address;
            row.port = this_port;
        });

        // Add a single stake to the metavars object.
        auto global = global_state.get_or_create(_self);
        global.total_stake += 1;
        global_state.set(global, this_user);
    }
}

// Unsubscribes an element from the metagraph.
void bittensor::unsubscribe( name this_user )
{
    eosio::print("unsubscribe");
    eosio::print(this_user);
    require_auth(this_user);
    metagraph graph(get_self(), get_code().value);
    auto iterator = graph.find(this_user.value);
    check(iterator != graph.end(), "Record does not exist");
    graph.erase(iterator);

    // Update total_stake.
    auto global = global_state.get();
    global.total_stake -= iterator->stake;
    global_state.set(global, this_user);

    // TODO(const): We need to add the balance back into the bittensor pool
    // and remove this user from the metagraph.
}

// Emits pending stake release to this node AND updates edge set.
// NOTE(const): The release is applied assuming the previous edge
// set was in place up until this block.
void bittensor::emit( const name this_user,
                      const std::vector<std::pair<name, float> > this_edges )

{
  eosio::print("emit");
  eosio::print(this_user);

  // Requires caller authority.
  require_auth( this_user );

  // (1) Assert this_id is subscribed.
  metagraph graph(get_self(), get_code().value);
  auto iterator = graph.find(this_user.value);
  check(iterator != graph.end(), "Error: Node is not subscribed");
  const auto& node = *iterator;
  uint64_t this_stake = node.stake;
  uint64_t this_last_emit = node.last_emit;

  // (2) Assert edge set length.
  int MAX_ALLOWED_EDGES = 3;
  if (this_edges.size() <= 0 || this_edges.size() > MAX_ALLOWED_EDGES) {
    check(false, "Error: Edge set length must be >= 0 and <= MAX_ALLOWED_EDGES");
  }

  // (3) Assert id is at position 0.
  if (this_edges.at(0).first.value != this_user.value) {
    check(false, "Error: First edge should point to self");
  }

  float sum = 0.0;
  auto edge_itr = this_edges.begin();
  for(;edge_itr != this_edges.end(); ++edge_itr) {
    sum += edge_itr->second;

    eosio::print('e:');
    eosio::print(edge_itr->first);
    eosio::print(edge_itr->second);

    // (4) Assert all weights > 0.0
    if (edge_itr->second < 0.0) {
      check(false, "Error: Edges should be > 0.0");
    }
  }
  // (5) Assert weight sum == 1.0
  if (sum != 1.0) {
    check(false, "Edges should sum to 1.0");
  }

  // (6) Calculate the Emission total.
  uint64_t this_emission = _get_emission(this_user, this_last_emit, this_stake);

  // (7) Apply emission.
  std::vector<std::pair<name, uint64_t> > emission_queue;
  emission_queue.push_back(std::make_pair(this_user, this_emission));

  while (emission_queue.size() > 0) {
    // Pop the next element.
    auto current = emission_queue.back();
    emission_queue.pop_back();
    const name current_user = current.first;
    const uint64_t current_emission = current.second;

    // Pull edge information.
    auto current_user_iterator = graph.find(current_user.value);
    if (current_user_iterator == graph.end()) {
      continue;
    }
    const auto& current_node = *current_user_iterator;
    const std::vector<std::pair<name, float> > current_edges = current_node.edges;

    // Emit to self.
    // NOTE(const): The assumption is that the inedge is stored at position 0.
    float current_inedge = current_edges.at(0).second;
    uint64_t current_stake  = current_node.stake;
    uint64_t stake_addition = (current_stake * current_inedge) + 1;
    graph.modify(current_user_iterator, this_user, [&]( auto& row ) {
      row.stake += stake_addition;
    });

    // Increment the total_stake by this emission quantity.
    auto global = global_state.get();
    global.total_stake += stake_addition;
    global_state.set(global, this_user);

    // Emit to neighbors.
    auto edge_itr = current_edges.begin();

    // NOTE(const) Ignore the self-edge which was previously applied.
    edge_itr++;

    for(;edge_itr != current_edges.end(); ++edge_itr) {

      // Calculate the emission along this edge.
      const name next_user= edge_itr->first;
      const float next_weight = edge_itr->second;
      const uint64_t next_emission = current_emission * next_weight;

      // Base case on zero emission. Can take a long time emission is large.
      // Fortunately each recursive call removes a token from the emission.
      if (next_emission > 0) {
        emission_queue.push_back(std::make_pair(next_user, next_emission));
      }
    }
  }


  // (8) Set new state
  graph.modify(iterator, this_user, [&]( auto& row ) {
    row.edges = this_edges;
    row.last_emit = tapos_block_num();
  });
}

uint64_t bittensor::_get_emission( const name this_user,
                                const uint64_t this_last_emit,
                                const uint64_t this_stake ) {
  // Constants for this emission system.
  const uint64_t BLOCKS_TILL_EMIT = 1;
  const float SUPPLY_EMIT_RATE = 1;
  const uint64_t total_stake = global_state.get().total_stake;

  // Calculate the number of blocks since this id's last emission.
  const uint64_t delta_blocks = this_last_emit - tapos_block_num();

  eosio::print("_get_emission");
  eosio::print(this_last_emit);
  eosio::print(this_stake);
  eosio::print(total_stake);
  eosio::print(delta_blocks);
  eosio::print(BLOCKS_TILL_EMIT);

  uint64_t this_emission;
  this_emission = SUPPLY_EMIT_RATE * delta_blocks * (this_stake / total_stake);

  return this_emission + 1;
}

void bittensor::_do_emit( const name this_user,
                          const uint64_t this_emission ) {

  eosio::print("_do_emit");
  eosio::print(this_user);
  eosio::print(this_emission);

  metagraph graph(get_self(), get_code().value);
  std::vector<std::pair<name, uint64_t> > emission_queue;
  emission_queue.push_back(std::make_pair(this_user, this_emission));

  while (emission_queue.size() > 0) {
    // Pop the next element.
    auto current = emission_queue.back();
    emission_queue.pop_back();
    const name current_user = current.first;
    const uint64_t current_emission = current.second;

    eosio::print("dequeue");
    eosio::print(current_user);
    eosio::print(current_emission);

    // Pull edge information.
    auto iterator = graph.find(current_user.value);
    if (iterator == graph.end()) {
      continue;
    }
    const auto& current_node = *iterator;
    const std::vector<std::pair<name, float> > current_edges = current_node.edges;

    // Emit to self.
    // NOTE(const): The assumption is that the inedge is stored at position 0.
    float current_inedge = current_edges.at(0).second;
    uint64_t current_stake  = current_node.stake;
    uint64_t stake_addition = current_stake * current_inedge + 1;
    graph.modify(iterator, current_user, [&]( auto& row ) {
      row.stake += stake_addition;
    });

    // Increment the total_stake by this emission quantity.
    auto global = global_state.get();
    global.total_stake += stake_addition;
    global_state.set(global, this_user);

    // Emit to neighbors.
    auto edge_itr = current_edges.begin();

    // NOTE(const) Ignore the self-edge which was previously applied.
    edge_itr++;

    for(;edge_itr != current_edges.end(); ++edge_itr) {

      // Calculate the emission along this edge.
      const name next_user= edge_itr->first;
      const float next_weight = edge_itr->second;
      const uint64_t next_emission = current_emission * next_weight;

      // Base case on zero emission. Can take a long time emission is large.
      // Fortunately each recursive call removes a token from the emission.
      if (next_emission > 0) {
        emission_queue.push_back(std::make_pair(next_user, next_emission));
      }
    }
  }
}

void bittensor::create( name   issuer,
                        asset  maximum_supply )
{
    require_auth( _self );

    auto sym = maximum_supply.symbol;
    check( sym.is_valid(), "invalid symbol name" );
    check( maximum_supply.is_valid(), "invalid supply");
    check( maximum_supply.amount > 0, "max-supply must be positive");

    stats statstable( _self, sym.code().raw() );
    auto existing = statstable.find( sym.code().raw() );
    check( existing == statstable.end(), "token with symbol already exists" );

    statstable.emplace( _self, [&]( auto& s ) {
       s.supply.symbol = maximum_supply.symbol;
       s.max_supply    = maximum_supply;
       s.issuer        = issuer;
    });
}


void bittensor::issue( name to, asset quantity, string memo )
{
    auto sym = quantity.symbol;
    check( sym.is_valid(), "invalid symbol name" );
    check( memo.size() <= 256, "memo has more than 256 bytes" );

    stats statstable( _self, sym.code().raw() );
    auto existing = statstable.find( sym.code().raw() );
    check( existing != statstable.end(), "token with symbol does not exist, create token before issue" );
    const auto& st = *existing;

    require_auth( st.issuer );
    check( quantity.is_valid(), "invalid quantity" );
    check( quantity.amount > 0, "must issue positive quantity" );

    check( quantity.symbol == st.supply.symbol, "symbol precision mismatch" );
    check( quantity.amount <= st.max_supply.amount - st.supply.amount, "quantity exceeds available supply");

    statstable.modify( st, same_payer, [&]( auto& s ) {
       s.supply += quantity;
    });

    add_balance( st.issuer, quantity, st.issuer );

    if( to != st.issuer ) {
      SEND_INLINE_ACTION( *this, transfer, { {st.issuer, "active"_n} },
                          { st.issuer, to, quantity, memo }
      );
    }
}

void bittensor::retire( asset quantity, string memo )
{
    auto sym = quantity.symbol;
    check( sym.is_valid(), "invalid symbol name" );
    check( memo.size() <= 256, "memo has more than 256 bytes" );

    stats statstable( _self, sym.code().raw() );
    auto existing = statstable.find( sym.code().raw() );
    check( existing != statstable.end(), "token with symbol does not exist" );
    const auto& st = *existing;

    require_auth( st.issuer );
    check( quantity.is_valid(), "invalid quantity" );
    check( quantity.amount > 0, "must retire positive quantity" );

    check( quantity.symbol == st.supply.symbol, "symbol precision mismatch" );

    statstable.modify( st, same_payer, [&]( auto& s ) {
       s.supply -= quantity;
    });

    sub_balance( st.issuer, quantity );
}

void bittensor::transfer( name    from,
                      name    to,
                      asset   quantity,
                      string  memo )
{
    check( from != to, "cannot transfer to self" );
    require_auth( from );
    check( is_account( to ), "to account does not exist");
    auto sym = quantity.symbol.code();
    stats statstable( _self, sym.raw() );
    const auto& st = statstable.get( sym.raw() );

    require_recipient( from );
    require_recipient( to );

    check( quantity.is_valid(), "invalid quantity" );
    check( quantity.amount > 0, "must transfer positive quantity" );
    check( quantity.symbol == st.supply.symbol, "symbol precision mismatch" );
    check( memo.size() <= 256, "memo has more than 256 bytes" );

    auto payer = has_auth( to ) ? to : from;

    sub_balance( from, quantity );
    add_balance( to, quantity, payer );
}

void bittensor::sub_balance( name owner, asset value ) {
   accounts from_acnts( _self, owner.value );

   const auto& from = from_acnts.get( value.symbol.code().raw(), "no balance object found" );
   check( from.balance.amount >= value.amount, "overdrawn balance" );

   from_acnts.modify( from, owner, [&]( auto& a ) {
         a.balance -= value;
      });
}

void bittensor::add_balance( name owner, asset value, name ram_payer )
{
   accounts to_acnts( _self, owner.value );
   auto to = to_acnts.find( value.symbol.code().raw() );
   if( to == to_acnts.end() ) {
      to_acnts.emplace( ram_payer, [&]( auto& a ){
        a.balance = value;
      });
   } else {
      to_acnts.modify( to, same_payer, [&]( auto& a ) {
        a.balance += value;
      });
   }
}

void bittensor::open( name owner, const symbol& symbol, name ram_payer )
{
   require_auth( ram_payer );

   auto sym_code_raw = symbol.code().raw();

   stats statstable( _self, sym_code_raw );
   const auto& st = statstable.get( sym_code_raw, "symbol does not exist" );
   check( st.supply.symbol == symbol, "symbol precision mismatch" );

   accounts acnts( _self, owner.value );
   auto it = acnts.find( sym_code_raw );
   if( it == acnts.end() ) {
      acnts.emplace( ram_payer, [&]( auto& a ){
        a.balance = asset{0, symbol};
      });
   }
}

void bittensor::close( name owner, const symbol& symbol )
{
   require_auth( owner );
   accounts acnts( _self, owner.value );
   auto it = acnts.find( symbol.code().raw() );
   check( it != acnts.end(), "Balance row already deleted or never existed. Action won't have any effect." );
   check( it->balance.amount == 0, "Cannot close because the balance is not zero." );
   acnts.erase( it );
}

} /// namespace eosio

EOSIO_DISPATCH( eosio::bittensor, (subscribe)(unsubscribe)(emit)(create)(issue)(transfer)(open)(close)(retire) )

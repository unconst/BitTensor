/**
 *  @file
 *  @copyright defined in eos/LICENSE.txt
 */

#include "bittensor.hpp"
#include "eosiolib/transaction.hpp"

namespace eosio {

// Subscribes a new account to the metagraph.
void bittensor::subscribe( const name user,
                           const std::string address,
                           const std::string port)
{

    // Require authority from the calling user.
    require_auth( user );
    peer_table ptable(get_self(), get_code().value);
    auto iterator = ptable.find(user.value);

    // Add a new element to the graph with 0 stake.
    if( iterator == ptable.end() )
    {
        // NOTE(const): Initially all nodes have a single edge to themselves.
        vector<pair<user, float> > this_edges;
        this_edges.push_back(make_pair(user, 1.0));

        // NOTE(const): We are emitting a single token on subscribe which opens up
        // potential sybil attacks. This may need to change, or protective measure
        // put into place.
        total_supply += 1;
        ptable.emplace(user, [&]( auto& row ) {
            row.identity = user;
            row.stake = 1;
            row.last_emit = tapos_block_num();
            row.edges = this_edges;
            row.address = address;
            row.port = port;
        });
    }
}

// Unsubscribes an element from the metagraph.
void bittensor::unsubscribe( name user )
{
    require_auth(user);
    peer_table ptable(get_self(), get_code().value);
    auto iterator = ptable.find(user.value);
    check(iterator != ptable.end(), "Record does not exist");
    ptable.erase(iterator);
}

// Emits pending stake release to this node AND updates edge set.
// NOTE(const): The release is applied assuming the previous edge
// set was in place up until this block.
void bittensor::emit(  const name this_user,
                       const vector<pair<name, float> > this_edges)

{
  // Requires caller authority.
  require_auth( user );

  // (1) Assert this_id is subscribed.
  peer_table ptable(get_self(), get_code().value);
  auto iterator = ptable.find(this_user.value);
  check(iterator != ptable.end(), "Error: Node is not subscribed");
  const auto& node = *iterator;
  asset this_stake = node.stake

  // (2) Assert edge set length.
  if (this_edges.size() <= 0 || this_edges.size() > MAX_ALLOWED_EDGES) {
    check(false, "Error: Edge set length must be >= 0 and <= MAX_ALLOWED_EDGES");
  }

  // (3) Assert id is at position 0.
  if (this_edges.at(0).second.value != this_user.value) {
    check(false, "Error: First edge should point to self");
  }

  float sum = 0.0;
  auto edge_itr = this_edges.begin();
  for(;edge_itr != this_edges.end(); ++edge_itr) {
    sum += edge_itr->second;
    // (4) Assert all weights > 0.0
    if (edge_itr->second > 0.0) {
      check(false, "Error: Edges should > 0.0");
    }
  }
  // (5) Assert weight sum == 1.0
  if (sum != 1.0) {
    check(false, "Edges should sum to 1.0");
  }

  // (6) Calculate the Emission total.
  asset this_emission = _get_emission(this_user, this_stake);

  // (7) Apply emission.
  _do_emit(this_user, this_emission);

  // (8) Set new state
  ptable.modify(iterator, this_user, [&]( auto& row ) {
    row.edges = this_edges;
    row.last_emit = tapos_block_num();
  });
}

unsigned int BitTensor::_get_emission(name this_user,
                                      const asset this_last_emit,
                                      const asset this_stake) {

  // Constants for this emission system.
  const unsigned int BLOCKS_TILL_EMIT = 1;
  const float SUPPLY_EMIT_RATE = 1;

  // Calculate the number of blocks since this id's last emission.
  const uint64_t delta_blocks = this_last_emit - tapos_block_num();

  unsigned int this_emission;
  this_emission = SUPPLY_EMIT_RATE * delta_blocks * (this_stake / total_supply);

  return this_emission;
}

void BitTensor::_do_emit(peer_table ptable,
                         const name this_user,
                         const asset this_emission) {

  vector<pair<name, asset> > emission_queue;
  emission_queue.push_back(make_pair(this_user, this_emission));

  while (emission_queue.size() > 0) {
    // Pop the next element.
    auto current = emission_queue.back();
    emission_queue.pop_back();
    const name current_user = current.first;
    const asset current_emission = current.second;

    // Pull edge information.
    auto iterator = ptable.find(current_user.value);
    if (this_edges_itr == ptable.end()) {
      continue;
    }
    const auto& current_node = *iterator;
    const vector<pair<name, float> > current_edges = node.edges;

    // Emit to self.
    // NOTE(const): The assumption is that the inedge is stored at position 0.
    float current_inedge = current_edges.at(0).second;
    asset current_stake  = current_node.stake;
    asset stake_addition = current_stake * current_inedge;
    ptable.modify(iterator, this_user, [&]( auto& row ) {
      row.stake = row.stake + stake_addition
    });
    total_supply += stake_addition;

    // Emit to neighbors.
    vector<pair<name float> >::iterator edge_itr = current_edges.begin();

    // NOTE(const) Ignore the self-edge which was previously applied.
    edge_itr++;

    for(;edge_itr != this_edges.end(); ++edge_itr) {

      // Calculate the emission along this edge.
      const name next_user= edge_itr->first;
      const float next_weight = edge_itr->second;
      const asset next_emission = current_emission.value * next_weight;

      // Base case on zero emission. Can take a long time emission is large.
      // Fortunately each recursive call removes a token from the emission.
      if (next_emission.value > 0) {
        emission_queue.push_back(make_pair(next_user, next_emission));
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

EOSIO_DISPATCH( eosio::bittensor, (upsert)(grade)(erase)(create)(issue)(transfer)(open)(close)(retire) )

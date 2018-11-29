from wenchain.core import Transaction
from wenchain.core import Block
from wenchain.core import BlockChain
from wenchain.core import TransactionPool
from wenchain.core import Wallet
from wenchain.miner.manager import Manager


class Miner(object):

    def __init__(self, wallet, block_chain, transaction_pool):
        self.wallet = wallet
        self.block_chain = block_chain
        self.transaction_pool = transaction_pool

    def mine(self):
        # 1. Add reward for the miner
        # 2. Create/mine block from valid transactions
        # 3. Sync the chain with other p2p servers (TODO: add after finishing proof of concept)
        # 4. Clear the local pool
        # 5. Broadcast that pool is clear to other nodes via p2p
        height = len(self.block_chain.chain) + 1
        valid_transactions = self.transaction_pool.get_valid_transactions()
        reward_transaction = Transaction.create_reward_transaction(height, self.wallet)
        valid_transactions.append(reward_transaction)
        block = self.block_chain.append_block(valid_transactions)
        # TODO: Broadcast Sync Chain action across p2p nodes
        self.transaction_pool.clear()
        # TODO: Broadcast Clear Transaction Pool action across p2p nodes
        return block

    def create_transaction(self, recipient_address, amount):
        return self.wallet.create_transaction(recipient_address, amount,
                                              self.transaction_pool, self.block_chain)

    def clear_transactions(self, session):
        manager = Manager(session)
        manager.clear_transactions()

    def save_transactions(self, session):
        manager = Manager(session)
        manager.save_transactions(self.transaction_pool.transactions)

    def save_blocks(self, session):
        manager = Manager(session)
        manager.save_blocks(self.block_chain.chain)

    def save_miner(self, session):
        manager = Manager(session)
        manager.save_wallet(self.wallet)
        manager.save_blocks(self.block_chain.chain)
        manager.save_transactions(self.transaction_pool.transactions)

    @classmethod
    def load_miner(cls, session, wallet_name):
        manager = Manager(session)
        wallet = manager.load_wallet(wallet_name)
        if wallet:
            chain = []
            for block in manager.load_blocks():
                chain.append(Block.from_database_format(block))
            transactions = []
            for transaction in manager.load_transactions():
                transactions.append(Transaction.from_database_format(transaction))
            miner = cls(
                Wallet.from_database_format(wallet),
                BlockChain(chain=chain),
                TransactionPool(transactions=transactions)
            )
            return miner
        return None

    @classmethod
    def create_miner(cls, wallet_password, wallet_name=None, wallet_balance=0):
        wallet = Wallet.create_wallet(wallet_password, name=wallet_name, balance=wallet_balance)
        block_chain = BlockChain()
        transaction_pool = TransactionPool()
        miner = cls(wallet, block_chain, transaction_pool)
        return miner

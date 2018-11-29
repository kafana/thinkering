import time
import unittest


from wenchain.core import TransactionPool
from wenchain.core import BlockChain
from wenchain.core import Wallet
from wenchain.miner import Miner
from wenchain.core.common import MINING_REWARD


TEST_WALLET_BALANCE = 500


class MinerTests(unittest.TestCase):

    password = b'P4sSw0rD'

    def test_multiple_miners(self):
        transaction_pool = TransactionPool()
        block_chain = BlockChain()

        wallet1 = Wallet.create_wallet(self.password, balance=TEST_WALLET_BALANCE)
        miner1 = Miner(wallet1, block_chain, transaction_pool)

        wallet2 = Wallet.create_wallet(self.password, balance=TEST_WALLET_BALANCE)
        miner2 = Miner(wallet2, block_chain, transaction_pool)

        # Send money from miner1 to miner2
        amount1 = 20
        wallet1.create_transaction(miner2.wallet.public_key_hex, amount1,
                                   transaction_pool, block_chain)

        # Send money from miner2 to miner1
        amount2 = 30
        wallet2.create_transaction(miner1.wallet.public_key_hex, amount2,
                                   transaction_pool, block_chain)

        # Mine new block (miner1)
        miner1.mine()

        # Pool should be clear and transactions should be included into the mined block
        self.assertTrue(len(transaction_pool.transactions) == 0)
        self.assertTrue(len(block_chain.chain) == 2)

        # Send money from miner2 to miner1
        amount3 = 3
        wallet2.create_transaction(miner1.wallet.public_key_hex, amount3,
                                   transaction_pool, block_chain)

        # Send money from miner1 to miner2
        amount4 = 100
        wallet1.create_transaction(miner2.wallet.public_key_hex, amount4,
                                   transaction_pool, block_chain)

        self.assertEqual(miner1.wallet.balance,
                         TEST_WALLET_BALANCE - amount1 + amount2 + MINING_REWARD)
        self.assertEqual(miner2.wallet.balance, TEST_WALLET_BALANCE + amount1 - amount2)

        # Mine new block (miner2)
        miner2.mine()
        time.sleep(2)

        # Pool should be clear and transactions should be included into the mined block
        self.assertTrue(len(transaction_pool.transactions) == 0)
        self.assertTrue(len(block_chain.chain) == 3)

        amount5 = 1
        wallet2.create_transaction(miner1.wallet.public_key_hex, amount5, transaction_pool,
                                   block_chain)

        amount6 = 1
        wallet1.create_transaction(miner2.wallet.public_key_hex, amount6, transaction_pool,
                                   block_chain)

        self.assertEqual(miner2.wallet.balance,
                         TEST_WALLET_BALANCE + amount1 -
                         amount2 - amount3 + amount4 + MINING_REWARD)
        self.assertEqual(miner1.wallet.balance,
                         TEST_WALLET_BALANCE - amount1 + amount2 +
                         MINING_REWARD + amount3 - amount4)

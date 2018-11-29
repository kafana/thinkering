import unittest
import copy


from wenchain.core import Transaction, TransactionInput, TransactionOutput
from wenchain.core import Block, BlockChain
from wenchain.core.common import ChainUtil


class BlockChainTests(unittest.TestCase):

    def create_test_transaction(self,
                                timestamp=None,
                                amount=0,
                                address='myaddress',
                                signature='mysignature'):
        if timestamp == None:
            timestamp = ChainUtil.generate_timestamp()
        transaction = Transaction(
            None,
            TransactionInput(timestamp, amount, address, signature),
            [TransactionOutput(amount, address)],
        )
        return transaction

    def test_append_block(self):
        block_chain = BlockChain()
        transaction1 = self.create_test_transaction(amount=1)
        transaction2 = self.create_test_transaction(amount=2)
        block_chain.append_block([transaction1, transaction2])
        self.assertEqual(len(block_chain.chain), 2)
        self.assertEqual(block_chain.chain[1].data[0].input.amount, 1)
        self.assertEqual(block_chain.chain[1].data[1].input.amount, 2)
        transaction3 = self.create_test_transaction(amount=3)
        transaction4 = self.create_test_transaction(amount=4)
        block_chain.append_block([transaction3, transaction4])
        self.assertEqual(len(block_chain.chain), 3)
        self.assertEqual(block_chain.chain[2].data[0].input.amount, 3)
        self.assertEqual(block_chain.chain[2].data[1].input.amount, 4)
        self.assertEqual(block_chain.chain[0].hash, block_chain.chain[1].prev_hash)
        self.assertEqual(block_chain.chain[1].hash, block_chain.chain[2].prev_hash)

    def test_compare_and_hash(self):
        block_chain1 = BlockChain()
        transaction = self.create_test_transaction()
        block_chain1.append_block([transaction,])
        block_chain2 = copy.deepcopy(block_chain1)
        self.assertEqual(block_chain1, block_chain2)
        self.assertEqual(hash(block_chain1), hash(block_chain2))
        block_chain3 = BlockChain()
        block_chain3.append_block([transaction,])
        self.assertNotEqual(block_chain1, block_chain3)
        self.assertNotEqual(hash(block_chain1), hash(block_chain3))

    def test_replace_chain(self):
        block_chain1 = BlockChain()
        transaction1 = self.create_test_transaction(amount=1)
        block_chain1.append_block([transaction1,])
        transaction2 = self.create_test_transaction(amount=2)
        block_chain1.append_block([transaction2,])
        block_chain2 = copy.deepcopy(block_chain1)
        transaction3 = self.create_test_transaction(amount=3)
        block_chain2.append_block([transaction3,])
        self.assertTrue(block_chain1.replace_chain(block_chain2))
        self.assertEqual(block_chain1, block_chain2)
        self.assertEqual(hash(block_chain1), hash(block_chain2))

        block_chain1 = BlockChain()
        transaction1 = self.create_test_transaction(amount=1)
        block_chain1.append_block([transaction1,])
        transaction2 = self.create_test_transaction(amount=2)
        block_chain1.append_block([transaction2,])
        block_chain2 = copy.deepcopy(block_chain1)
        self.assertFalse(block_chain1.replace_chain(block_chain2))

    def test_valid_chain(self):
        block_chain1 = BlockChain()
        transaction1 = self.create_test_transaction(amount=1)
        block_chain1.append_block([transaction1,])
        transaction2 = self.create_test_transaction(amount=2)
        block_chain1.append_block([transaction2,])
        block_chain2 = copy.deepcopy(block_chain1)
        transaction3 = self.create_test_transaction(amount=3)
        block_chain2.append_block([transaction3,])
        self.assertTrue(block_chain1.is_valid(block_chain2))

        block_chain2.chain[1].hash = 'wrong hash'
        self.assertFalse(block_chain1.is_valid(block_chain2))

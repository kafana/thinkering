import unittest


from wenchain.core import Transaction, TransactionInput, TransactionOutput
from wenchain.core.block import Block, MINE_DIFFICULTY
from wenchain.core.common import ChainUtil


class BlockTests(unittest.TestCase):

    amount = 0
    address = 'myaddress'
    signature = 'mysignature'

    def create_test_transaction(self, timestamp=None):
        if timestamp == None:
            timestamp = ChainUtil.generate_timestamp()
        transaction = Transaction(
            None,
            TransactionInput(timestamp, self.amount, self.address, self.signature),
            [TransactionOutput(self.amount, self.address)],
        )
        return transaction

    def test_mine(self):
        genesis = Block.genesis()
        transaction = self.create_test_transaction()
        block = Block.mine(genesis, [transaction])
        self.assertEqual(block.data[0].outputs[0].amount, self.amount)
        self.assertEqual(block.data[0].outputs[0].address, self.address)
        self.assertEqual(block.data[0].input.amount, self.amount)
        self.assertEqual(block.data[0].input.address, self.address)
        self.assertEqual(genesis.hash, block.prev_hash)

    def test_compare_and_hash(self):
        genesis = Block.genesis()
        transaction = self.create_test_transaction()
        block1 = Block.mine(genesis, [transaction,])
        block2 = Block(block1.timestamp, block1.prev_hash,
                       block1.hash, block1.nonce,
                       block1.difficulty, block1.data)
        self.assertEqual(block1, block2)
        self.assertEqual(hash(block1), hash(block2))
        transaction = self.create_test_transaction()
        block3 = Block.mine(genesis, [transaction,])
        self.assertNotEqual(block1, block3)
        self.assertNotEqual(hash(block1), hash(block3))

    def test_difficulty(self):
        transaction = self.create_test_transaction()
        genesis = Block.genesis()
        block1 = Block.mine(genesis, [transaction,])
        # Genesis has old timestamp, so this shouldn't fail
        self.assertTrue(block1.hash.startswith('0' * MINE_DIFFICULTY))

    def test_difficulty_level(self):
        transaction = self.create_test_transaction()
        genesis = Block.genesis()
        block = Block.mine(genesis, [transaction,])
        block1 = Block.mine(block, [transaction,])

        timestamp = block1.timestamp + ChainUtil.to_milliseconds(hours=1)
        difficulty = Block.calculate_difficulty(block1, timestamp)
        self.assertNotEqual(difficulty, block1.difficulty)
        self.assertTrue(difficulty < block1.difficulty)

        timestamp = block1.timestamp - ChainUtil.to_milliseconds(hours=1)
        difficulty = Block.calculate_difficulty(block1, timestamp)
        self.assertNotEqual(difficulty, block1.difficulty)
        self.assertTrue(difficulty > block1.difficulty)

    def _test_generate_genesis_block(self):
        pb = Block(ChainUtil.generate_timestamp(), None,
                   '0000000000000000000000000000000000000000000000000000000000000000', 0, 3, [])
        address = ('000000000000000000000000000000000000000000000000000'
                   '0000000000000556e757365642067656e6573697320626c6f63'
                   '6b2061646472657373202d2031312f30332f323031382032333'
                   'a3036204553540000000000')
        signature = '47656e6573697320626c6f636b2063726561746564206f6e2031312f30332f32303138'
        timestamp = 1541319432152
        transaction = Transaction(
            None,
            TransactionInput(timestamp, amount, address, signature),
            [TransactionOutput(amount, address)],
        )
        block = Block.mine(pb, [transaction])

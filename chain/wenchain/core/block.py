import sys
import json
import hashlib
from copy import deepcopy


from wenchain.core import (Transaction,
                           TransactionInput,
                           TransactionOutput)
from wenchain.core.common import (
    ChainUtil,
    REWARD_TRANSACTION_ADDRESS,
    REWARD_TRANSACTION_SIGNATURE)


MINE_DIFFICULTY = 3 # Number of leading hash 0s
MINE_RATE = 10000 # Milliseconds


class Block(object):

    def __init__(self, timestamp, prev_hash, hash, nonce, difficulty, data):
        self.timestamp = timestamp
        self.prev_hash = prev_hash
        self.hash = hash
        self.nonce = nonce
        self.difficulty = difficulty or MINE_DIFFICULTY
        self.data = data

    def __repr__(self):
        return "<Block(timestamp=%r, prev_hash=%r, hash=%r, nonce=%r, difficulty=%r, data=%r)>" % (
            self.timestamp, self.prev_hash, self.hash, self.nonce, self.difficulty, self.data)

    def __eq__(self, other):
        if isinstance(other, Block):
            return self.hash == other.hash
        return False

    def __hash__(self):
        return hash(self.hash)

    def __copy__(self):
        return type(self)(
            self.timestamp,
            self.prev_hash,
            self.hash,
            self.nonce,
            self.difficulty,
            self.data)

    # The ids param is a dict of id's to copies
    # memoization avoids recursion
    def __deepcopy__(self, ids):
        self_id = id(self)
        item = ids.get(self_id)
        if item is None:
            item = type(self)(
                deepcopy(self.timestamp, ids),
                deepcopy(self.prev_hash, ids),
                deepcopy(self.hash, ids),
                deepcopy(self.nonce, ids),
                deepcopy(self.difficulty, ids),
                deepcopy(self.data, ids))
            ids[self_id] = item
        return item

    @property
    def serialize(self):
        return {
            'timestamp': self.timestamp,
            'prev_hash': self.prev_hash,
            'hash': self.hash,
            'nonce': self.nonce,
            'difficulty': self.difficulty,
            'data': [transaction.serialize for transaction in self.data]
        }

    def is_reward_transaction_valid(self, height):
        # Check if block contains single reward transaction.
        # We also flag blocks as valid if they don't have reward transactions.
        valid = True
        reward_counter = 0
        amount = ChainUtil.get_block_value(height)
        for transaction in self.data:
            if (transaction.input.address == REWARD_TRANSACTION_ADDRESS or
                    transaction.input.signature == REWARD_TRANSACTION_SIGNATURE):
                reward_counter += 1
                if len(transaction.outputs) != 1 or transaction.outputs[0].amount != amount:
                    valid = False
                    continue
        # Make sure we don't have mre than 1 reward transaction.
        if reward_counter > 1:
            valid = False
        return valid

    @classmethod
    def from_database_format(cls, item):
        transactions = json.loads(item.data)
        return cls(item.timestamp,
                   item.prev_hash,
                   item.hash,
                   item.nonce,
                   item.difficulty,
                   [Transaction.from_database_format(data) for data in transactions])

    @classmethod
    def genesis(cls):
        # 0000ef80bfa3b57331b1200969d8044267ce1af1a4d2980308e2b6af4569ff33
        timestamp = 1541320453883
        nonce = 46645
        difficulty = 4
        prev_hash = '0000000000000000000000000000000000000000000000000000000000000000'
        address = ('00000000000000000000000000000000000000000000'
                   '00000000000000000000556e757365642067656e6573'
                   '697320626c6f636b2061646472657373202d2031312f'
                   '30332f323031382032333a3036204553540000000000')
        signature = '47656e6573697320626c6f636b2063726561746564206f6e2031312f30332f32303138'
        amount = 0
        transaction = Transaction(
            '7f5f7b62dfe211e89b13685b35ad2541',
            TransactionInput(1541319432152, amount, address, signature),
            [TransactionOutput(amount, address)]
        )
        data = [transaction]
        hash = Block.create_hash(timestamp, prev_hash, nonce, difficulty, data)
        return cls(timestamp, prev_hash, hash, nonce, difficulty, data)

    @classmethod
    def mine(cls, previous_block, data):
        timestamp = None
        difficulty = None
        nonce = 0
        for nonce in range(nonce, sys.maxsize):
            timestamp = ChainUtil.generate_timestamp()
            difficulty = cls.calculate_difficulty(previous_block, timestamp)
            hash = Block.create_hash(timestamp, previous_block.hash, nonce, difficulty, data)
            if hash.startswith('0' * difficulty):
                break
        return cls(timestamp, previous_block.hash, hash, nonce, difficulty, data)

    @classmethod
    def calculate_difficulty(cls, block, timestamp):
        """
        Raise difficulty level by 1 if last block was mined too
        fast, otherwise lower difficulty level by 1
        """
        # Figure out if we need to change difficulty level
        delta = block.timestamp + MINE_RATE
        difficulty = block.difficulty + 1 if delta > timestamp else block.difficulty - 1
        # Make sure we never go under minimum difficulty level
        if difficulty < MINE_DIFFICULTY:
            difficulty = MINE_DIFFICULTY
        return difficulty

    @staticmethod
    def create_block_hash(block):
        return Block.create_hash(block.timestamp, block.prev_hash, block.nonce,
                                 block.difficulty, block.data)

    @staticmethod
    def create_hash(timestamp, prev_hash, nonce, difficulty, data):
        m = hashlib.sha256()
        # TODO: Avoid collision
        # TODO: Implement Merkle trees
        for transaction in data:
            m.update(transaction.create_digest())
        m.update(str(timestamp).encode('ascii'))
        m.update(prev_hash.encode('ascii'))
        m.update(str(nonce).encode('ascii'))
        m.update(str(difficulty).encode('ascii'))
        return m.hexdigest()

import logging
from copy import deepcopy


from wenchain.core.block import Block


logger = logging.getLogger(__name__)


class BlockChain(object):

    def __init__(self, chain=None):
        self.chain = chain if chain != None else [Block.genesis()]

    def __repr__(self):
        return "<BlockChain(chain=%r)>" % (
            self.chain)

    def __eq__(self, other):
        if not isinstance(other, BlockChain):
            return False
        for i in range(len(self.chain)):
            if self.chain[i] != other.chain[i]:
                return False
        return True

    def __hash__(self):
        return hash(tuple([block.hash for block in self.chain]))

    def __copy__(self):
        return type(self)(self.chain)

    # The ids param is a dict of id's to copies
    # memoization avoids recursion
    def __deepcopy__(self, ids):
        self_id = id(self)
        item = ids.get(self_id)
        if item is None:
            item = type(self)(deepcopy(self.chain, ids))
            ids[self_id] = item
        return item

    def append_block(self, data):
        previous_block = self.chain[-1]
        block = Block.mine(previous_block, data)
        self.chain.append(block)
        return block

    def replace_chain(self, block_chain):
        if len(block_chain.chain) <= len(self.chain):
            return False

        if not self.is_valid(block_chain):
            return False

        self.chain = deepcopy(block_chain.chain)

        return True

    def is_valid(self, block_chain):
        # Compare existing chain with new chain
        for i in range(0, len(self.chain)):
            if (self.chain[i].hash != block_chain.chain[i].hash or
                    self.chain[i].prev_hash != block_chain.chain[i].prev_hash):
                return False

        # Check additional blocks of the new chain
        for j in range(i, len(block_chain.chain)):
            prev_block = block_chain.chain[i - 1]
            block = block_chain.chain[i]
            if prev_block.hash != block.prev_hash or Block.create_block_hash(block) != block.hash:
                return False
            if not block.is_reward_transaction_valid(j):
                logger.error('Invalid reward transaction for block {}'.format(block.hash))
                return False

        return True

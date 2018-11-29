import math
import json
import uuid
import hashlib
from collections import OrderedDict


from wenchain.core.common import (
    ChainUtil,
    REWARD_TRANSACTION_ADDRESS,
    REWARD_TRANSACTION_SIGNATURE)


class TransactionInput(object):

    def __init__(self, timestamp, amount, address, signature):
        self.timestamp = timestamp
        self.amount = amount
        self.address = address
        self.signature = signature

    def __repr__(self):
        return "<TransactionInput(timestamp=%r, amount=%r, address=%r, signature=%r)>" % (
            self.timestamp, self.amount, self.address, self.signature)

    @property
    def serialize(self):
        return {
            'timestamp': self.timestamp,
            'amount': self.amount,
            'address': self.address,
            'signature': self.signature,
        }

    def to_database_format(self):
        return self.serialize

    @classmethod
    def from_database_format(cls, item):
        return cls(item['timestamp'], item['amount'], item['address'], item['signature'])


class TransactionOutput(object):

    def __init__(self, amount, address):
        self.amount = amount
        self.address = address

    def __repr__(self):
        return "<TransactionOutput(amount=%r, address=%r)>" % (
            self.amount, self.address)

    @property
    def serialize(self):
        return {
            'amount': self.amount,
            'address': self.address
        }

    def to_database_format(self):
        return self.serialize

    @classmethod
    def from_database_format(cls, item):
        return cls(item['amount'], item['address'])


class Transaction(object):

    def __init__(self, transaction_id, input, outputs):
        self.transaction_id = transaction_id or uuid.uuid1().hex
        self.input = input
        self.outputs = outputs

    def __repr__(self):
        return "<Transaction(transaction_id=%r, input=%r, outputs=%r)>\n\n" % (
            self.transaction_id, self.input, self.outputs)

    @property
    def serialize(self):
        return {
            'transaction_id': self.transaction_id,
            'input': self.input.serialize,
            'outputs': [output.serialize for output in self.outputs]
        }

    def create_outputs_digest(self):
        m = hashlib.sha256()
        for output in self.outputs:
            m.update(json.dumps(output.serialize, sort_keys=True).encode('utf8'))
        return m.digest()

    def create_digest(self):
        m = hashlib.sha256()
        m.update(self.transaction_id.encode('utf8'))
        for output in self.outputs:
            m.update(json.dumps(output.serialize, sort_keys=True).encode('utf8'))
        m.update(json.dumps(self.input.serialize, sort_keys=True).encode('utf8'))
        return m.digest()

    def verify(self):
        return ChainUtil.verify_signature(
            bytes.fromhex(self.input.signature),
            bytes.fromhex(self.input.address),
            self.create_outputs_digest())

    def sign(self, senders_wallet):
        """
        Every time we call sign method we recreate input dict.
        """
        self.input = TransactionInput(
            ChainUtil.generate_timestamp(),
            senders_wallet.balance,
            senders_wallet.public_key_hex,
            senders_wallet.sign(self.create_outputs_digest()))

    def update(self, senders_wallet, recipient_address, amount):
        for output in self.outputs:
            if output.address == senders_wallet.public_key_hex:
                # Disallow transactions where amount is greater than original
                # output amount (out of balance)
                # We do this so we don't endup with -amount balance
                if amount > output.amount:
                    raise ValueError('Invalid update amount. Exiting amount '
                                     'of {} is greater than new amount of {}'.format(
                                         output.amount, amount))
                # Update balance
                output.amount = output.amount - amount
                break
        self.outputs.append(TransactionOutput(amount, recipient_address)) # To
        self.sign(senders_wallet)

    @classmethod
    def from_database_format(cls, item):
        if isinstance(item, dict):
            input = TransactionInput.from_database_format(item['input'])
            outputs = [TransactionOutput.from_database_format(output) for output in item['outputs']]
            return cls(item['transaction_id'], input, outputs)
        else:
            input = TransactionInput.from_database_format(json.loads(item.input))
            outputs = [TransactionOutput.from_database_format(output)
                       for output in json.loads(item.outputs)]
            return cls(item.transaction_id, input, outputs)

    @classmethod
    def create_transaction(cls, senders_wallet, recipient_address, amount):
        if amount > senders_wallet.balance:
            raise ValueError('Invalid wallet balance')

        outputs = [
            TransactionOutput(senders_wallet.balance - amount,
                              senders_wallet.public_key_hex), # From
            TransactionOutput(amount, recipient_address), # To
        ]

        transaction = cls(uuid.uuid1().hex, None, outputs)
        transaction.sign(senders_wallet)

        return transaction

    @classmethod
    def create_reward_transaction(cls, height, miner_wallet):
        # There are two main differences between regular transaction and reward transaction
        # 1. Outputs contain single entry
        # 2. Inputs are not signed
        # 3. Each block can only contain single reward transaction with single output entry

        outputs = [
            TransactionOutput(ChainUtil.get_block_value(height), miner_wallet.public_key_hex), # To
        ]

        input = TransactionInput(
            ChainUtil.generate_timestamp(), 0,
            REWARD_TRANSACTION_ADDRESS, REWARD_TRANSACTION_SIGNATURE
        )

        return cls(uuid.uuid1().hex, input, outputs)


class TransactionPool(object):

    def __init__(self, transactions=None):
        self.transactions = OrderedDict()
        if transactions:
            if isinstance(transactions, dict):
                for key, value in transactions.items():
                    self.transactions[key] = value
            elif isinstance(transactions, (list, tuple,)):
                for transaction in transactions:
                    self.transactions[transaction.transaction_id] = transaction

    def __repr__(self):
        return "<TransactionPool(transactions=%r)>" % (
            self.transactions)

    def add(self, transaction):
        self.transactions[transaction.transaction_id] = transaction

    def get_valid_transactions(self):
        # 1. Total output amount matches the original balance specified in the input balance
        # 2. Verify signature of every transaction
        transactions = []
        for key, transaction in self.transactions.items():
            total = 0
            for output in transaction.outputs:
                total += output.amount

            if transaction.input.amount != total:
                print('Invalid transaction from {}. Output '
                      'amount doesn\'t match input amounts'.format(transaction.input.address))
                continue
                # TODO: Log this
                # raise ValueError('Invalid transaction from {}. Output '
                #     'amount doesn\'t match input amounts'.format(transaction.input.address))
            if not transaction.verify():
                print('Invalid transaction from {}. Invalid '
                      'signature'.format(transaction.input.address))
                continue
                # TODO: Log this
                # raise ValueError('Invalid transaction from {}. Invalid '
                #     'signature'.format(transaction.input.address))

            transactions.append(transaction)

        return transactions

    def find_by_public_key(self, public_key_hex):
        for key, value in self.transactions.items():
            if value.input.address == public_key_hex:
                return value
        return None

    def clear(self):
        self.transactions.clear()

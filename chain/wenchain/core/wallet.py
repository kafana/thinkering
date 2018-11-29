import logging


from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


from wenchain.core import Transaction
from wenchain.core.common import ChainUtil


logger = logging.getLogger(__name__)


class Wallet(object):

    def __init__(self, password, private_key_pem, public_key_pem,
                 balance=0, name=None,
                 last_processed_block_hash=None):
        self.password = password
        self.private_key_pem = private_key_pem
        self.public_key_pem = public_key_pem
        self.balance = balance
        self._private_key = key = serialization.load_pem_private_key(
            self.private_key_pem,
            password=self.password,
            backend=default_backend()
        )
        self._public_key = serialization.load_pem_public_key(
            self.public_key_pem,
            backend=default_backend()
        )
        self.name = name or self.public_key_hex
        # This is used for balance calculation
        self.last_processed_block_hash = last_processed_block_hash

    @property
    def serialize(self):
        return {
            'password': '****************', # TODO: Security issue
            'private_key_pem': self.private_key_pem, # TODO: Security issue
            'public_key_hex': self.public_key_hex,
            'public_key_pem': self.public_key_pem,
            'name': self.name,
            'last_processed_block_hash': self.last_processed_block_hash
        }

    def __repr__(self):
        return "<Wallet(balance=%r, public_key_pem=%r, name=%r)>" % (
            self.balance, self.public_key_pem, self.name)

    @classmethod
    def from_database_format(cls, item):
        return cls(item.password, item.private_key_pem,
                   item.public_key_pem, balance=item.balance,
                   name=item.name,
                   last_processed_block_hash=item.last_processed_block_hash)

    @property
    def public_key_der(self):
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo)

    @property
    def public_key_hex(self):
        return self.public_key_der.hex()

    def sign(self, data):
        signature = self._private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )
        return signature.hex()

    def verify(self, signature, data):
        return ChainUtil.verify_signature(signature, self.public_key_der, data)

    def create_transaction(self, recipient_address, amount, transaction_pool, block_chain):
        self.balance = self.calculate_balance(block_chain)

        if amount > self.balance:
            raise ValueError('Amount {} exceeds wallet balance of {}'.format(amount, self.balance))

        # ##########################################################################################
        # Optimization code
        # We make sure that multiple TransactionOutput objects are appended to existing transactions
        # ##########################################################################################
        transaction = transaction_pool.find_by_public_key(self.public_key_hex)

        if transaction:
            transaction.update(self, recipient_address, amount)
        else:
            transaction = Transaction.create_transaction(self, recipient_address, amount)
            transaction_pool.add(transaction)
        # ##########################################################################################

        return transaction


    def calculate_balance(self, block_chain):
        balance = self.balance
        transactions = []

        # Move all unprocessed transactions to an array
        if not self.last_processed_block_hash:
            for block in block_chain.chain:
                transactions.extend([transaction for transaction in block.data])
        else:
            start = False
            for block in block_chain.chain:
                if start:
                    transactions.extend([transaction for transaction in block.data])
                    continue
                if block.hash == self.last_processed_block_hash:
                    start = True

        self.last_processed_block_hash = block_chain.chain[-1].hash

        # 1. Filter out all transactions that are created by/belong to this wallet
        #    and subtract or add TransactionOutput amount from the wallet balance.
        #    We'll also subtract payments to other wallets and add rewards from mining, if any.
        for transaction in transactions:
            if transaction.input.address == self.public_key_hex:
                for output in transaction.outputs:
                    if output.address != self.public_key_hex:
                        balance -= output.amount
            else:
                for output in transaction.outputs:
                    if output.address == self.public_key_hex:
                        balance += output.amount

        return balance

    @classmethod
    def create_wallet(cls, password, balance=0, name=None):
        private_key, public_key = Wallet.generate_keys(password)
        return cls(password, private_key, public_key, balance=balance, name=name)

    @staticmethod
    def generate_keys(password,
                      encoding=serialization.Encoding.PEM,
                      private_key_format=serialization.PrivateFormat.PKCS8,
                      public_key_format=serialization.PublicFormat.SubjectPublicKeyInfo):
        # ec.SECP256K1() is used by bitcoin
        private_key = ec.generate_private_key(ec.SECP384R1(), default_backend())
        serialized_public = private_key.public_key().public_bytes(
            encoding=encoding,
            format=public_key_format
        )
        serialized_private = private_key.private_bytes(
            encoding=encoding,
            format=private_key_format,
            encryption_algorithm=serialization.BestAvailableEncryption(password)
        )

        return serialized_private, serialized_public

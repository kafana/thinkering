import math
from datetime import datetime, timedelta


from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature


COIN_REWARD = 50 # coins
COIN_FRACTIONS = 100 * 1000 * 1000 # fractions
MINING_REWARD = COIN_REWARD * COIN_FRACTIONS
NUMBER_OF_BLOCKS_BEFORE_SPLIT = 210000

REWARD_TRANSACTION_ADDRESS = 'REWARD-ADDRESS'
REWARD_TRANSACTION_SIGNATURE = 'REWARD-SIGNATURE'

class ChainUtil(object):

    @staticmethod
    def verify_signature(signature, public_key, data):
        try:
            serialization.load_der_public_key(
                public_key,
                backend=default_backend()
            ).verify(signature, data, ec.ECDSA(hashes.SHA256()))
        except InvalidSignature as exc:
            return False
        return True

    @staticmethod
    def generate_timestamp():
        return math.floor(datetime.utcnow().timestamp() * 1000)

    @staticmethod
    def to_datetime(timestamp):
        return datetime.fromtimestamp(timestamp / 1000.0)

    @staticmethod
    def to_milliseconds(**kwargs):
        delta = timedelta(**kwargs)
        return (delta.days * (24 * 60 * 60 * 1000) +
                delta.seconds * 1000 +
                math.floor(delta.microseconds / 1000))

    @staticmethod
    def get_block_value(height, fees=0):
        subsidy = MINING_REWARD
        subsidy >>= math.floor(height / NUMBER_OF_BLOCKS_BEFORE_SPLIT)
        return subsidy + fees

    @staticmethod
    def get_coin_decimal_value(coins):
        return coins / float(COIN_FRACTIONS)

    @staticmethod
    def _get_maximum_chain_value():
        fees = 0
        total = 0
        height = 0
        subsidy_and_fees = MINING_REWARD + fees
        while subsidy_and_fees != 0:
            subsidy_and_fees = ChainUtil.get_block_value(height, fees=fees)
            height += 1
            total += subsidy_and_fees
        return total / float(COIN_FRACTIONS)

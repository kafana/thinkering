import json
import logging


from sqlalchemy.orm.exc import NoResultFound


from wenchain.db.models import TransactionModel
from wenchain.db.models import BlockModel
from wenchain.db.models import WalletModel
from wenchain.db.common import ModelUtil


logger = logging.getLogger(__name__)


class Manager(object):

    def __init__(self, session, **kwargs):
        self.session = session

    def load_entity(self, klass, **kwargs):
        query = self.session.query(klass)
        query = ModelUtil.build_query_filters(klass, query, **kwargs)
        try:
            return query.one()
        except NoResultFound:
            return None

    def load_collection(self, klass, **kwargs):
        query = self.session.query(klass)
        query = ModelUtil.build_query_filters(klass, query, **kwargs)
        for item in query.yield_per(10).enable_eagerloads(False):
            yield item

    def load_transactions(self, **kwargs):
        for transaction in self.load_collection(TransactionModel):
            yield transaction

    def save_transactions(self, transactions):
        def _update(entity, transaction):
            entity.transaction_id = transaction.transaction_id
            entity.input = json.dumps(transaction.serialize['input'])
            entity.outputs = json.dumps(transaction.serialize['outputs'])
            return entity
        # Make sure we refresh existing transaction, if found in db
        for _, transaction in transactions.items():
            try:
                entity = self.session.query(TransactionModel).filter(
                    TransactionModel.transaction_id == transaction.transaction_id).one()
            except NoResultFound:
                entity = TransactionModel()
                self.session.add(entity)
            entity = _update(entity, transaction)
            self.session.commit()

    def load_blocks(self, **kwargs):
        for block in self.load_collection(BlockModel):
            yield block

    def save_blocks(self, blocks):
        def _update(entity, block):
            entity.timestamp = block.timestamp
            entity.prev_hash = block.prev_hash
            entity.hash = block.hash
            entity.nonce = block.nonce
            entity.difficulty = block.difficulty
            entity.data = json.dumps(block.serialize['data'])
            return entity
        # No need to save block that's already in db
        for block in blocks:
            try:
                entity = self.session.query(BlockModel).filter(
                    BlockModel.hash == block.hash).one()
            except NoResultFound:
                entity = BlockModel()
                entity = _update(entity, block)
                self.session.add(entity)
                self.session.commit()

    def load_wallet(self, wallet_name):
        return self.load_entity(WalletModel, **{'name': wallet_name})

    def save_wallet(self, wallet):
        def _update(entity, wallet):
            entity.name = wallet.name
            entity.password = wallet.password
            entity.private_key_pem = wallet.private_key_pem
            entity.public_key_pem = wallet.public_key_pem
            entity.last_processed_block_hash = wallet.last_processed_block_hash
            entity.balance = wallet.balance
            return entity
        try:
            entity = self.session.query(WalletModel).filter(
                WalletModel.name == wallet.name).one()
            entity = _update(entity, wallet)
        except NoResultFound:
            entity = WalletModel()
            entity = _update(entity, wallet)
            self.session.add(entity)
        self.session.commit()

    def clear_transactions(self):
        try:
            rows_deleted = self.session.query(TransactionModel).delete()
            self.session.commit()
            logger.info('Removed {} rows from transactions table'.format(rows_deleted))
        except:
            self.session.rollback()

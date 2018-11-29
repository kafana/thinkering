import os
import logging


from sqlalchemy import Column, Boolean, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base


logger = logging.getLogger(__name__)


_Model = declarative_base()


class Base(_Model):

    __abstract__ = True

    id = Column(Integer, primary_key=True)
    created = Column(DateTime, default=func.current_timestamp())
    modified = Column(DateTime, default=func.current_timestamp(),
                      onupdate=func.current_timestamp())
    active = Column(Boolean, nullable=False, default=True)


class WalletModel(Base):
    __tablename__ = 'wallets'

    name = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=False)
    private_key_pem = Column(String, nullable=False)
    public_key_pem = Column(String, nullable=False)
    balance = Column(Integer, nullable=False)
    last_processed_block_hash = Column(String, nullable=True)

    def __repr__(self):
        return "<WalletModel(name='%s')>" % (self.name)


class BlockModel(Base):
    __tablename__ = 'blocks'

    timestamp = Column(Integer, nullable=False)
    prev_hash = Column(String, nullable=False, unique=True)
    hash = Column(String, nullable=False, unique=True)
    nonce = Column(Integer, nullable=False)
    difficulty = Column(Integer, nullable=False)
    data = Column(Text, nullable=False)

    def __repr__(self):
        return "<BlockModel(hash='%s')>" % (self.hash)


class TransactionModel(Base):
    __tablename__ = 'transactions'

    transaction_id = Column(String, nullable=False, unique=True)
    input = Column(Text, nullable=False)
    outputs = Column(Text, nullable=False)

    def __repr__(self):
        return "<TransactionModel(transaction_id='%s')>" % (self.transaction_id)

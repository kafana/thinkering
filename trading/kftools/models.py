import os
import warnings
import logging

import sqlite3
from sqlalchemy import Table, Column, Boolean, Integer, String, DateTime, ForeignKey, Float, Date, cast
from sqlalchemy import create_engine, UniqueConstraint, event, exc, inspect
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.sql.expression import asc as asc_expr, desc as desc_expr


logger = logging.getLogger(__name__)


class Base(declarative_base()):

    __abstract__ = True

    id = Column(Integer, primary_key=True)
    created = Column(DateTime, default=func.current_timestamp())
    modified = Column(DateTime, default=func.current_timestamp(),
                      onupdate=func.current_timestamp())
    active = Column(Boolean, nullable=False, default=True)


companies_tags_association = Table('companies_tags', Base.metadata,
    Column('company_id', Integer, ForeignKey('companies.id')),
    Column('tag_id', Integer, ForeignKey('tags.id'))
)


companies_sectors_association = Table('companies_sectors', Base.metadata,
    Column('company_id', Integer, ForeignKey('companies.id')),
    Column('sector_id', Integer, ForeignKey('sectors.id'))
)


class Industry(Base):
    __tablename__ = 'industries'

    name = Column(String, nullable=False, unique=True)
    companies = relationship("Company", back_populates="industry")

    def __repr__(self):
        return "<Industry(name='%s')>" % (self.name)

    @property
    def serialize_terminal(self):
        return "{}\t{}".format(self.id, self.name)


class Company(Base):
    __tablename__ = 'companies'

    symbol = Column(String, nullable=False, unique=True)
    exchange = Column(String, nullable=True, index=True)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    website = Column(String, nullable=True)
    ceo = Column(String, nullable=True)
    issue_type = Column(String, nullable=False)
    industry_id = Column(Integer, ForeignKey('industries.id'))
    industry = relationship("Industry", back_populates="companies")
    tags = relationship("Tag",
                        secondary=companies_tags_association,
                        back_populates="companies")
    sectors = relationship("Sector",
                           secondary=companies_sectors_association,
                           back_populates="companies")

    def __repr__(self):
        return "<Company(symbol='%s', name='%s')>" % (
            self.symbol, self.name)

    @property
    def serialize_terminal(self):
        return "{}\t{}\t{}".format(self.id, self.symbol, self.name)


class Tag(Base):
    __tablename__ = 'tags'

    name = Column(String, unique=True)
    companies = relationship("Company",
                             secondary=companies_tags_association,
                             back_populates="tags")

    def __repr__(self):
        return "<Tag(name='%s')>" % (self.name)

    @property
    def serialize_terminal(self):
        return "{}\t{}".format(self.id, self.name)


class Sector(Base):
    __tablename__ = 'sectors'

    name = Column(String, unique=True)
    companies = relationship("Company",
                             secondary=companies_sectors_association,
                             back_populates="sectors")

    def __repr__(self):
        return "<Sector(name='%s')>" % (self.name)

    @property
    def serialize_terminal(self):
        return "{}\t{}".format(self.id, self.name)


class ImpliedVolatility(Base):
    __tablename__ = 'implied_volatility'
    __table_args__ = (UniqueConstraint('symbol', 'day', 'term', name='_symbol_day_term_uc'),)

    day = Column(DateTime, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    call = Column(Float, nullable=False)
    put = Column(Float, nullable=False)
    term = Column(String, nullable=False, index=True)
    call_open_interest = Column(Integer, nullable=False)
    put_open_interest = Column(Integer, nullable=False)

    def __repr__(self):
        return "<ImpliedVolatility(symbol=%r, day=%r, term=%r, call_open_interest=%r, put_open_interest=%r)>" % (
            self.symbol, self.day, self.term, self.call_open_interest, self.put_open_interest)

    @property
    def serialize_terminal(self):
        return "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(self.id, self.day,
                                                       self.symbol, self.call,
                                                       self.put, self.term,
                                                       self.put_open_interest,
                                                       self.call_open_interest)

    @property
    def serialize_data_frame(self):
        data = {}
        data['day'] = self.day
        data['symbol'] = self.symbol
        data['call'] = self.call
        data['put'] = self.put
        data['term'] = self.term
        data['call_open_interest'] = self.call_open_interest
        data['put_open_interest'] = self.put_open_interest
        return data


def add_engine_pidguard(engine):
    """Add multiprocessing guards.

    Forces a connection to be reconnected if it is detected
    as having been shared to a sub-process.

    """
    logger.info('Enabled sqlalchemy multiprocessing database engine guard')

    @event.listens_for(engine, "connect")
    def connect(dbapi_connection, connection_record):
        connection_record.info['pid'] = os.getpid()

    @event.listens_for(engine, "checkout")
    def checkout(dbapi_connection, connection_record, connection_proxy):
        pid = os.getpid()
        if connection_record.info['pid'] != pid:
            logger.warn(
                "Parent process %(orig)s forked (%(newproc)s) with an open "
                "database connection, "
                "which is being discarded and recreated." %
                {"newproc": pid, "orig": connection_record.info['pid']})
            connection_record.connection = connection_proxy.connection = None
            raise exc.DisconnectionError(
                "Connection record belongs to pid %s, "
                "attempting to check out in pid %s" %
                (connection_record.info['pid'], pid)
            )


class DatabaseClient(object):

    def __init__(self, db_uri='sqlite:///:memory:',
                       enable_engine_pidguard=False, **kwargs):
        self.engine = None
        self.session_maker = None
        self._session = None
        self.init(db_uri, enable_engine_pidguard, **kwargs)

    def init(self, db_uri='sqlite:///:memory:', enable_engine_pidguard=False, **kwargs):
        self.engine = create_engine(db_uri, **kwargs)
        if enable_engine_pidguard:
            add_engine_pidguard(self.engine)
        self.session_maker = scoped_session(sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine))

    @property
    def session(self):
        if not self._session:
            self._session = self.session_maker()
        return self._session

    @session.deleter
    def session(self):
        if self._session:
            self.session_maker.remove()
            self._session = None

    def create_schema(self):
        Base.metadata.create_all(self.engine)

    @staticmethod
    def is_sqlite_db(file):
        try:
            conn = sqlite3.connect('file:{}?mode=ro'.format(file), uri=True)
            conn.close()
            return True
        except sqlite3.OperationalError as exc:
            logger.warn('File {} is not valid sqlite3 database'.format(file),
                        exc_info=exc)
        return False


class Manager(object):

    def __init__(self, init_schema=True, **kwargs):
        self.client = DatabaseClient(**kwargs)
        if init_schema:
            self._init_db_schema()


    def _init_db_schema(self):
        self.client.create_schema()


    def create_iex_company(self, data):
        try:
            company = self.client.session.query(Company).filter(
                Company.symbol==data['symbol']).one()
            company.name = data['companyName']
            company.exchange = data['exchange']
            company.issue_type = data['issueType']
            company.description = data['description']
            company.website = data['website']
            company.ceo = data['CEO']
        except NoResultFound:
            company = Company(
                symbol=data['symbol'],
                name=data['companyName'], 
                exchange=data['exchange'],
                issue_type=data['issueType'],
                description=data['description'],
                website=data['website'],
                ceo=data['CEO'])
            self.client.session.add(company)

        # Tags
        if 'tags' in data:
            for name in data['tags']:
                tag = self._create_generic(Tag, name)
                company.tags.append(tag)
        # Industry
        if 'industry' in data and data['industry']:
            industry_name = data['industry']
        else:
            if data['issueType'] == 'et':
                industry_name = 'ETF'
            else:
                industry_name = 'Unknown'
        company.industry = self._create_generic(Industry, industry_name)
        # Sector(s)
        if 'sector' in data and data['sector']:
            sector_name = data['sector']
        else:
            if data['issueType'] == 'et':
                sector_name = 'ETF'
            else:
                sector_name = 'Unknown'
        company.sectors.append(self._create_generic(Sector, sector_name))

        self.client.session.commit()

        return company


    def create_industry(self, name):
        return self._create_generic(Industry, name, True)


    def create_tag(self, name):
        return self._create_generic(Tag, name, True)


    def create_sector(self, name):
        return self._create_generic(Sector, name, True)


    def _create_generic(self, klass, name, commit_session=False):
        try:
            entity = self.client.session.query(klass).filter(
                klass.name==name).one()
        except NoResultFound:
            entity = klass(name=name)
            self.client.session.add(entity)
            if commit_session:
                self.client.session.commit()
        return entity


    def get_companies_by_tag(self, **kwargs):
        query = self.client.session.query(Tag)
        try:
            return self._build_query_filters(Tag, query, **kwargs).one().companies
        except NoResultFound:
            pass
        return []


    def get_companies_by_sector(self, **kwargs):
        query = self.client.session.query(Sector)
        try:
            return self._build_query_filters(Sector, query, **kwargs).one().companies
        except NoResultFound:
            pass
        return []


    def get_companies_by_industry(self, **kwargs):
        query = self.client.session.query(Company).join(Industry)
        return self._build_query_filters(Industry, query, **kwargs).all()


    def fetch_collection_for_class(self, klass, order_by='id',
                                   is_asc=True, limit=100, **kwargs):
        query = self.client.session.query(klass)
        return self._build_query_filters(klass, query, order_by=order_by,
                                         is_asc=is_asc, limit=limit, **kwargs).all()


    def _build_query_filters(self, klass, query, order_by='id',
                             is_asc=True, limit=100, **kwargs):
        order_by_expressions = []
        order_by_items = order_by if isinstance(order_by, list) else [order_by]

        # Setup filter expressions
        for key, value in kwargs.items():
            for column in inspect(klass).columns:
                if column.name == key:
                    try:
                        field = getattr(klass, key)
                    except AttributeError:
                        break

                    if isinstance(value, str) and '%' in value:
                        query = query.filter(field.like(value))
                    else:
                        query = query.filter(field==value)
                    break

        # Setup order by expressions
        for name in order_by_items:
            for column in inspect(klass).columns:
                if column.name == name:
                    try:
                        expr = getattr(klass, name)
                    except AttributeError:
                        break
                    if is_asc:
                        order_by_expressions.append(
                            asc_expr(expr))
                    else:
                        order_by_expressions.append(
                            desc_expr(expr))
                    break

        return query.order_by(
            *order_by_expressions).limit(limit)
        # return query.order_by(
        #    *order_by_expressions).limit(limit).offset(offset)


    def get_company_by_symbol(self, symbol):
        try:
            return self.client.session.query(Company).filter(
                Company.symbol==symbol).one()
        except NoResultFound:
            pass
        return None


    def get_iv_by_symbol_and_term(self, symbol, term):
        return self.client.session.query(ImpliedVolatility).filter(
            (ImpliedVolatility.symbol==symbol) &
            (ImpliedVolatility.term==term)).all()


    def get_iv_symbols(self):
        return self.client.session.query(ImpliedVolatility.symbol).group_by(
            ImpliedVolatility.symbol).all()


    def create_iv(self, data):
        try:
            query = self.client.session.query(ImpliedVolatility).filter(
                (ImpliedVolatility.symbol==data['symbol']) &
                (ImpliedVolatility.day == data['day']) &
                (ImpliedVolatility.term==data['term']))
            entity = query.one()
            entity.call = data['call']
            entity.put = data['put']
            entity.put_open_interest = data['put_open_interest']
            entity.call_open_interest = data['call_open_interest']
        except NoResultFound:
            entity = ImpliedVolatility(
                symbol=data['symbol'],
                day=data['day'], 
                call=data['call'],
                put=data['put'],
                term=data['term'],
                put_open_interest = data['put_open_interest'],
                call_open_interest = data['call_open_interest'])
            self.client.session.add(entity)

        self.client.session.commit()

        return entity

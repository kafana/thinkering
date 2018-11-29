import logging


from sqlalchemy import inspect
from sqlalchemy.sql.expression import asc as asc_expr, desc as desc_expr


logger = logging.getLogger(__name__)


class ModelUtil(object):

    @staticmethod
    def build_query_filters(klass, query, order_by=None,
                            is_asc=True, limit=0, **kwargs):
        if order_by:
            order_by_items = order_by if isinstance(order_by, list) else [order_by]
        else:
            order_by_items = []

        # Setup filter expressions
        for key, value in kwargs.items():
            for column in inspect(klass).columns:
                if column.name == key:
                    try:
                        field = getattr(klass, key)
                    except AttributeError:
                        logger.error('Missing attribute {} from class {}'.format(
                            key, klass))
                    else:
                        if isinstance(value, str) and '%' in value:
                            query = query.filter(field.like(value))
                        else:
                            query = query.filter(field == value)
                    break

        # Setup order by expressions
        if order_by_items:
            order_by_expressions = []
            for name in order_by_items:
                for column in inspect(klass).columns:
                    if column.name == name:
                        try:
                            expr = getattr(klass, name)
                        except AttributeError:
                            logger.error('Missing attribute {} from class {}'.format(
                                key, klass))
                        else:
                            if is_asc:
                                order_by_expressions.append(
                                    asc_expr(expr))
                            else:
                                order_by_expressions.append(
                                    desc_expr(expr))
                        break
            query = query.order_by(
                *order_by_expressions).limit(limit)

        # Setup limit expression
        if limit > 0:
            query = query.limit(limit)

        return query

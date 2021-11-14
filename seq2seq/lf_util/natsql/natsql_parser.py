from semparse.natsql.natsql_v1_3 import inference_sql, Args
import sqlite3
import sys
from spider_evaluation.process_sql import get_tables_with_alias, Schema, get_schema
from nltk import word_tokenize
from spider_evaluation.process_sql import parse_sql as parse_sql_original
from spider_evaluation.process_sql import tokenize as tokenize_original

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except', 'distinct')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = (
'not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'not in', 'not like', 'not like',
'join')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or', 'except_', 'intersect_', 'union_', 'sub')  # COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


class Schema_Star(Schema):
    def __init__(self, schema):
        Schema.__init__(self, schema)
        # super(Schema_Star, self).__init__(schema)

    def _map(self, schema):
        idMap = {}  # {'*': "__all__"}
        for key, vals in schema.iteritems() if sys.version_info < (3, 0) else schema.items():
            for val in vals:
                idMap[key.lower() + '.' + val.lower()] = key.lower() + '.' + val.lower()
            idMap[key.lower() + '.*'] = key.lower() + '.*'

        for key in schema:
            idMap[key.lower()] = key.lower()

        return idMap


# class Args():
#     def __init__(self):
#         self.in_sub_sql = True
#         self.eq_sub_sql = False
#         self.keep_top_order_by = True


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs) - 1, -1, -2):
        qidx1 = quote_idxs[i - 1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2 + 1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2 + 1:]
        vals[key] = val

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if
               tok == "=" or tok == "in" or tok == "like"]  # make 'not in' and 'not like' together
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx - 1]
        if pre_tok in prefix:
            toks = toks[:eq_idx - 1] + [pre_tok + "="] + toks[eq_idx + 1:]
        elif pre_tok == 'not':
            toks = toks[:eq_idx - 1] + [pre_tok + " " + toks[eq_idx]] + toks[
                                                                        eq_idx + 1:]  # make 'not in' and 'not like' together
    return toks


def tokenize_nSQL(nsql, star_name, sepearte_star_name=True):
    toks = tokenize(nsql)
    idx_star = 0
    remove_idx = []
    for idx, tok in enumerate(toks):
        if tok in ['except', 'intersect', 'union']:
            toks[idx] = toks[idx] + '_'
        if tok == "@":  # or tok == "@@@":
            toks[idx] = '@.@'
        if sepearte_star_name:
            if tok == '*':
                toks[idx] = star_name[idx_star].lower() + ".*"
                idx_star += 1
        else:
            if tok == '*':
                toks[idx] = toks[idx - 2] + toks[idx - 1] + toks[idx]
                remove_idx.append(idx - 2)
                remove_idx.append(idx - 1)
    remove_offset = 0
    for remove in remove_idx:
        del toks[remove - remove_offset]
        remove_offset += 1
    return toks


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    if tok == "@.@":
        return start_idx + 1, "@.@"

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = tables_with_alias[alias] + "." + col
        return start_idx + 1, schema.idMap[key]

    assert default_tables is not None and len(default_tables) > 0, "Default tables should not be None or empty"

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema[table]:
            key = table + "." + tok
            return start_idx + 1, schema.idMap[key]

    assert False, "Error col: {}".format(tok)


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] in AGG_OPS:
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        assert idx < len_ and toks[idx] == '('
        idx += 1
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, [agg_id, col_id, isDistinct]

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, [agg_id, col_id, isDistinct]


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, [unit_op, col_unit1, col_unit2]


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx + 1] == "as":
        idx += 3
    else:
        idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif "\"" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    else:
        try:
            val = float(toks[idx])
            if str(val) != toks[idx]:
                val = int(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')' \
                    and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[
                end_idx] not in COND_OPS:  # toks[end_idx] != 'sub' and toks[end_idx] != 'or':
                end_idx += 1

            if toks[start_idx] in AGG_OPS and toks[end_idx] == ')':  # Add for where column = agg(column)
                end_idx += 1

            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        assert toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []

    if toks[idx] in COND_OPS:
        conds.append(toks[idx])
        idx += 1  # skip and/or

    while idx < len_:
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1

        assert idx < len_ and toks[idx] in WHERE_OPS, "Error condition: idx: {}, tok: {}".format(idx, toks[idx])
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            assert toks[idx] == 'and'
            idx += 1
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
        else:  # normal case: single value
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            val2 = None

        conds.append([not_op, op_id, val_unit, val1, val2])

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append([agg_id, val_unit])
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, [isDistinct, val_units]


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    assert 'from' in toks[start_idx:], "'from' not found"

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append([TABLE_TYPE['sql'], sql])
        else:
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append([TABLE_TYPE['table_unit'], table_unit])
            default_tables.append(table_name)
        if idx < len_ and toks[idx] == 'join':
            idx += 1  # skip join
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            conds.extend(this_conds)

        if isBlock:
            assert toks[idx] == ')'
            idx += 1
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc'  # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    assert toks[idx] == 'by'
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, [order_type, val_units]


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        return idx, int(toks[idx - 1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False  # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, start_idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    if isBlock:
        assert toks[idx] == ')'
        idx += 1  # skip ')'

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def check_sql_make_sense(sql, allowed_distinct_in_having=False, allowed_distinct_in_orderBy=False):
    if sql is None:
        return False

    # Select columns are same
    if len(sql['select'][1]) > 1:
        for i, icol in enumerate(sql['select'][1]):
            for j, jcol in enumerate(sql['select'][1]):
                if i != j:
                    if icol[1][1][1] == jcol[1][1][1] and icol[1][1][0] == jcol[1][1][0] and icol[0] == jcol[0]:
                        return False

    # Select * with other columns
    # Select distinct *
    if len(sql['select'][1]) > 1 or sql['select'][0]:
        for col in sql['select'][1]:
            if col[1][1][1] == '__all__' and col[1][1][0] == 0 and col[0] == 0:
                return False

    # distinct *
    for col in sql['select'][1]:
        if col[1][1][2] and col[1][1][1] == '__all__':
            return False

    # 1. where condition appearing * = value.
    #    Actually, if * appearing in where condition is not allowed.
    # 2. where condition appearing "distinct".
    for item in sql['where']:
        if item not in COND_OPS:
            if item[2][1][1] == '__all__' or item[2][1][2]:
                return False

    # 1. having condition appearing "distinct".
    # But actually, there are some distinct in having
    if not allowed_distinct_in_having:
        for item in sql['having']:
            if item not in COND_OPS:
                if item[2][1][2]:
                    return False

    # 1. order by *
    # 2. orderBy appearing "distinct". But actually, there are some distinct in orderBy
    if sql['orderBy']:
        for col in sql['orderBy'][1]:
            if col[1][1] == '__all__' and col[1][0] == 0 or not allowed_distinct_in_orderBy and col[1][2]:
                return False

    return True


def create_sql_from_natSQL(nsql, db_name, db, table_json, args=Args()):
    # Schema_Num MODEL:
    find_table = None
    for table in table_json:
        if table['db_id'] == db_name:
            find_table = table
            break

    # Schema MODEL:
    schema = Schema_Star(get_schema(db, find_table))

    nsql = nsql.replace(" .*", ".*")
    toks = tokenize_nSQL(nsql, None, False)

    table_name = None
    contain_from = False
    for i, tok in enumerate(toks):
        if tok == 'value':
            toks[i] = '"value"'
        elif not table_name and '.' in tok:
            table_name = tok.split('.')[0]
        elif tok == 'from':
            contain_from = True

    if not contain_from:
        for i, tok in enumerate(toks):
            if tok in ['where', 'order', 'group']:
                toks.insert(i, table_name)
                toks.insert(i, 'from')
                break
            if i == len(toks) - 1:
                toks.insert(i + 1, table_name)
                toks.insert(i + 1, 'from')
                break

    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    try:
        final_sql = None
        p_nsql = None
        run_step_record = 0
        _, p_nsql = parse_sql(toks, 0, tables_with_alias, schema)
        run_step_record = 1
        final_sql = inference_sql(p_nsql, find_table, args)

        toks = tokenize_original(final_sql)
        schema = Schema(get_schema(db, find_table))
        tables_with_alias = get_tables_with_alias(schema.schema, toks)
        _, test_grammar = parse_sql_original(toks, 0, tables_with_alias, schema)

        if check_sql_make_sense(test_grammar):
            return final_sql, p_nsql

    except:
        final_sql = None
        p_nsql = None
        pass

    return None, None
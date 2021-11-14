CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists', 'not in', 'not like')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

AND_OR_OPS = ('and', 'or', 'except_', 'intersect_', 'union_', 'sub')  # COND_OPS = ('and', 'or')
ALL_COND_OPS = ('and', 'or', 'except_', 'intersect_', 'union_', 'sub')
SPECIAL_COND_OPS = ('except_', 'intersect_', 'union_', 'sub')

SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')


def col_unit_back(col_unit, tables_with_alias=None):
    """
        tables_with_alias is needed when col_unit is T1.column not table.column
    """
    if col_unit == None:
        return None

    bool_agg = False
    col = ""

    if col_unit[2] != None and col_unit[2]:
        col = " distinct "
    if col_unit[0] > 0:
        col = AGG_OPS[col_unit[0]] + '(' + col
        bool_agg = True

    name = col_unit[1]
    if name.endswith("__"):
        name = name[:-2]
    if name.startswith("__"):
        name = name[2:]
    if name == 'all':
        name = '*'
    if name.endswith("*"):
        name = '*'

    nameArray = name.split('.')
    if len(nameArray) == 2 and tables_with_alias:
        table_name = nameArray[0]
        for key, value in tables_with_alias.items():
            if key != table_name and value == table_name:
                name = key + "." + nameArray[1]
                break

    col = col + name

    if bool_agg:
        col = col + ')'

    return col


def val_unit_back(val_unit, tables_with_alias=None):
    val = ""

    col_1 = col_unit_back(val_unit[1], tables_with_alias)
    col_2 = col_unit_back(val_unit[2], tables_with_alias)

    if val_unit[0] > 0 and col_2 != None:
        val = val + col_1 + UNIT_OPS[val_unit[0]] + col_2
    else:
        val = val + col_1

    return val


def select_unit_back(val_unit, tables_with_alias=None):
    val = ""
    if val_unit[0] > 0:  # agg
        val = AGG_OPS[val_unit[0]] + '('

    val += val_unit_back(val_unit[1], tables_with_alias)

    if val_unit[0] > 0:
        val = val + ')'

    return val


def col_unit_contain_agg(col_unit, tables_with_alias=None):
    if col_unit == None:
        return False
    if col_unit[0] > 0:
        return True
    return False


def val_unit_contain_agg(val_unit, tables_with_alias=None):
    return col_unit_contain_agg(val_unit[1])





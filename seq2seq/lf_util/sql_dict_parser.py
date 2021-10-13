import json

WHERE_OPS = (
    "NOT",
    "BETWEEN",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "IN",
    "LIKE",
    "IS",
    "EXISTS",
)
UNIT_OPS = ("none", "-", "+", "*", "/")
AGG_OPS = ("none", "MAX", "MIN", "COUNT", "SUM", "AVG")
TABLE_TYPE = {
    "sql": "sql",
    "table_unit": "table_unit",
}

COND_OPS = ("AND", "OR")
SQL_OPS = ("INTERSECT", "UNION", "EXCEPT")
ORDER_OPS = ("DESC", "ASC")


class SqlParser:
    def __init__(self, table_path):
        tables = json.load(open(table_path))
        self.tables = {x['db_id']: x for x in tables}
        self.current_table = None

    def unparse(self, db_id, sql_dict):
        self.current_table = self.tables[db_id]
        return self._unparse(sql_dict)

    def _unparse(self, sql_dict):
        select_clause = self._unparse_select(sql_dict['select'])
        from_clause = self._unparse_from(sql_dict['from'])
        where_clause = groupby_clause = having_clause = orderby_clause = limit_clause = ''
        intersect_clause = except_clause = union_clause = ''
        if sql_dict['where']:
            where_clause = self._unparse_where(sql_dict['where'])
        if sql_dict['groupBy']:
            groupby_clause = self._unparse_groupby(sql_dict['groupBy'])
        if sql_dict['having']:
            having_clause = self._unparse_having(sql_dict['having'])
        if sql_dict['orderBy']:
            orderby_clause = self._unparse_orderby(sql_dict['orderBy'])
        if sql_dict['limit']:
            limit_clause = self._unparse_limit(sql_dict['limit'])
        if sql_dict['intersect']:
            intersect_clause = 'INTERSECT ' + self._unparse(sql_dict['intersect'])
        if sql_dict['except']:
            except_clause = 'EXCEPT ' + self._unparse(sql_dict['except'])
        if sql_dict['union']:
            union_clause = 'UNION ' + self._unparse(sql_dict['union'])
        sql = ' '.join([x for x in [select_clause, from_clause, where_clause,
                                    groupby_clause, having_clause, orderby_clause, limit_clause,
                                    intersect_clause, except_clause, union_clause] if x != ''])
        return sql

    def _unparse_select(self, _sel):
        is_distinct = _sel[0]
        sel_list = []
        for sel_item in _sel[1]:
            agg_id, val_unit = sel_item
            unit_op, col_unit1, col_unit2 = val_unit
            sel_item_str = self._unparse_col_unit(col_unit1)
            if unit_op != 0:
                print('Warning: calculation between columns are used')
                sel_item2_str = self._unparse_col_unit(col_unit2)
                sel_item_str = ' '.join([sel_item_str, UNIT_OPS[unit_op], sel_item2_str])
            if agg_id > 0:
                sel_item_str = f'{AGG_OPS[agg_id]}({sel_item_str})'
            if is_distinct:
                sel_item_str = 'DISTINCT ' + sel_item_str
            sel_list.append(sel_item_str)
        return 'SELECT ' + ', '.join(sel_list)

    def _unparse_from(self, _from):
        table_units = _from['table_units']
        conds = _from['conds']
        table_unit_str_list = []
        for table_unit in table_units:
            table_type, table_id_or_sql = table_unit
            if table_type == 'table_unit':
                table_unit_str_list.append(self.current_table['table_names_original'][table_id_or_sql])
            else:
                table_unit_str_list.append(self._unparse(table_id_or_sql))
        cond_str_list = self._unparse_condition(conds, return_list=True)
        assert all(x != 'or' for x in cond_str_list)
        cond_str_list = [x for x in cond_str_list if x not in ('and', 'or')]
        # assert len(table_unit_str_list) == len(cond_str_list) + 1  # assertion on number of join condition
        str_segs = [table_unit_str_list[0]]
        for i in range(1, len(table_unit_str_list)):
            str_segs.append('JOIN')
            str_segs.append(table_unit_str_list[i])
        if cond_str_list:
            str_segs.append('ON')
            str_segs.append(cond_str_list[0])
            for i in range(1, len(cond_str_list)):
                str_segs.append('AND')
                str_segs.append(cond_str_list[i])
        return 'FROM ' + ' '.join(str_segs)

    def _unparse_where(self, _where):
        clause = 'WHERE ' + self._unparse_condition(_where)
        return clause

    def _unparse_groupby(self, _groupby):
        gb_str_list = []
        for gb_item in _groupby:
            gb_str = self._unparse_col_unit(gb_item)
            gb_str_list.append(gb_str)
        return 'GROUP BY ' + ', '.join(gb_str_list)

    def _unparse_orderby(self, _orderby):
        order_op_str = _orderby[0]
        val_unit_str_list = []
        for val_unit in _orderby[1]:
            unit_op, col_unit1, col_unit2 = val_unit
            col_unit_str = self._unparse_col_unit(col_unit1)
            if unit_op != 0:
                print('Warning: calculation between columns are used')
                col_unit2_str = self._unparse_col_unit(col_unit2)
                col_unit_str = ' '.join([col_unit_str, UNIT_OPS[unit_op], col_unit2_str])
            val_unit_str_list.append(col_unit_str)
        clause = 'ORDER BY ' + ', '.join(val_unit_str_list) + ' ' + order_op_str
        return clause

    def _unparse_having(self, _having):
        clause = 'HAVING ' + self._unparse_condition(_having)
        return clause

    def _unparse_limit(self, limit):
        return 'LIMIT ' + str(limit)

    def _unparse_col_unit(self, col_unit):
        agg_id, col_id, is_distinct = col_unit
        clause = ''
        table_id, column_name = self.current_table['column_names_original'][col_id]
        if table_id >= 0:
            column_name = self.current_table['table_names_original'][table_id] + '.' + column_name
        clause += column_name
        if agg_id > 0:
            clause = AGG_OPS[agg_id] + ' ' + clause
        if is_distinct:
            clause = 'DISTINCT ' + clause
        return clause

    def _unparse_condition(self, condition, return_list=False):
        cond_str_list = []
        for cond_unit in condition:
            if cond_unit in ('and', 'or'):
                cond_str_list.append(cond_unit)
            else:
                #cond unit
                not_op, op_id, val_unit, val1, val2 = cond_unit
                op_str = WHERE_OPS[op_id]
                # val_unit
                unit_op, col_unit1, col_unit2 = val_unit
                col_unit_str = self._unparse_col_unit(col_unit1)
                if unit_op != 0:
                    print('Warning: calculation between columns are used')
                    unit_op_str = UNIT_OPS[unit_op]
                    col_unit2_str = self._unparse_col_unit(col_unit2)
                    col_unit_str = ' '.join([col_unit_str, unit_op_str, col_unit2_str])
                val1_str = self._unparse_val(val1)
                val2_str = self._unparse_val(val2)
                if not_op:
                    assert op_str in ('in', 'like')  # todo: check here
                    op_str = 'NOT ' + op_str
                if 'between' not in op_str.lower():
                    cond_str_list.append(f'{col_unit_str} {op_str} {val1_str}')
                else:
                    assert op_str.lower() == 'between'
                    cond_str_list.append(f'{col_unit_str} {op_str} {val1_str} AND {val2_str}')
        if return_list is False:
            return ' '.join(cond_str_list)
        else:
            return cond_str_list

    def _unparse_val(self, val):
        if val is None:
            return None
        if isinstance(val, str):
            val_str = val
        elif isinstance(val, dict):
            val_str = self._unparse(val)
        elif isinstance(val, int) or isinstance(val, float):
            val_str = str(val)
        else:
            val_str = self._unparse_col_unit(val)
        return val_str


if __name__ == '__main__':
    parser = SqlParser('data/cosql/tables.json')
    dev_data = json.load(open('data/cosql/train.json'))
    for i in range(len(dev_data)):
        print(i)
        for j in range(len(dev_data[i]['interaction'])):
            case_db = dev_data[i]['database_id']
            case = dev_data[i]['interaction'][j]['sql']
            sql = dev_data[i]['interaction'][j]['query']
            print(sql)
            print(parser.unparse(case_db, case))
            print()

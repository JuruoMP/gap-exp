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


class SqlDictParser:
    def __init__(self, table_path, lower_column=False):
        tables = json.load(open(table_path))
        self.tables = {x['db_id']: x for x in tables}
        self.current_table = None
        self.current_sql_dict = None
        self.current_id = None
        self.lower_column = lower_column

    def unparse(self, i, db_id, sql_dict):
        # self.check_assertion(sql_dict)
        self.current_id = i
        self.current_table = self.tables[db_id]
        self.current_sql_dict = sql_dict
        sql = self._unparse(sql_dict)[1:-1].replace('.*', ' .*')
        sql = ' '.join(sql.split()).strip()
        return sql

    def _unparse(self, sql_dict):
        select_clause = self._unparse_select(sql_dict['select'])
        # from_clause = self._unparse_from(sql_dict['from'], include_join=True)
        where_clause = groupby_clause = having_clause = orderby_clause = join_clause = limit_clause = ''
        intersect_clause = except_clause = union_clause = ''

        def combine_where(where_clause, having_clause):
            ret = []
            if where_clause == '':
                if having_clause:
                    ret += [having_clause]
            else:
                ret += [where_clause, having_clause]
            return ret

        if sql_dict['where']:
            where_clause, join_clause = self._unparse_where(sql_dict['where'])
        # if sql_dict['groupBy']:
        #     groupby_clause = self._unparse_groupby(sql_dict['groupBy'])
        if sql_dict['having']:
            having_clause = self._unparse_having(sql_dict['having'])

        clause_list = [select_clause]
        if combine_where(where_clause, having_clause):
            clause_list += ['WHERE'] + combine_where(where_clause, having_clause)
        if sql_dict['intersect']:
            intersect_clause, intersect_join = self._unparse_where(sql_dict['intersect']['where'])
            # assert intersect_join == join_clause
            if combine_where(intersect_clause, intersect_join):
                clause_list += ['INTERSECT'] + combine_where(intersect_clause, intersect_join)
        if sql_dict['except']:
            except_clause, except_join = self._unparse_where(sql_dict['except']['where'])
            # assert except_join == join_clause
            if combine_where(except_clause, except_join):
                clause_list += ['EXCEPT'] + combine_where(except_clause, except_join)
        if sql_dict['union']:
            union_clause, union_join = self._unparse_where(sql_dict['union']['where'])
            # assert union_join == join_clause
            if combine_where(union_clause, union_join):
                clause_list += ['UNION'] + combine_where(union_clause, union_join)

        if sql_dict['orderBy']:
            orderby_clause = self._unparse_orderby(sql_dict['orderBy'])
        if sql_dict['limit']:
            limit_clause = self._unparse_limit(sql_dict['limit'])

        clause_list += [join_clause, orderby_clause, limit_clause]
        sql = ' '.join(clause_list)
        return '( ' + sql + ' )'

    def check_assertion(self, sql_dict):
        if sql_dict['intersect']:
            assert sql_dict['intersect']['select'] == sql_dict['select']
        if sql_dict['union']:
            assert sql_dict['union']['select'] == sql_dict['select']
        if sql_dict['except']:
            assert sql_dict['except']['select'] == sql_dict['select']

    def _unparse_select(self, _sel):
        is_distinct = _sel[0]
        sel_list = []
        for sel_item in _sel[1]:
            agg_id, val_unit = sel_item
            unit_op, col_unit1, col_unit2 = val_unit
            # todo: can default_table_id be a sql? should use the first one?
            if self.current_sql_dict['from']['table_units'][0][0] == 'sql':
                sub_sql_dict = self.current_sql_dict['from']['table_units'][0][1]
                default_table_id = sub_sql_dict['from']['table_units'][0][1]
            else:
                default_table_id = self.current_sql_dict['from']['table_units'][0][1]
            sel_item_str = self._unparse_col_unit(col_unit1, default_table_id)
            if unit_op != 0:
                print('Warning: calculation between columns are used')
                sel_item2_str = self._unparse_col_unit(col_unit2, default_table_id)
                sel_item_str = ' '.join([sel_item_str, UNIT_OPS[unit_op], sel_item2_str])
            if agg_id > 0:
                sel_item_str = f'{AGG_OPS[agg_id]} ( {sel_item_str} )'
            if is_distinct:
                sel_item_str = 'DISTINCT ' + sel_item_str
            sel_list.append(sel_item_str)
        return 'SELECT ' + ', '.join(sel_list)

    # def _unparse_from(self, _from, include_join=True):
    #     table_units = _from['table_units']
    #     conds = _from['conds']
    #     table_unit_str_list = []
    #     for table_unit in table_units:
    #         table_type, table_id_or_sql = table_unit
    #         if table_type == 'table_unit':
    #             table_name = self.current_table['table_names_original'][table_id_or_sql]
    #             table_unit_str_list.append(table_name if not self.lower_column else table_name.lower())
    #         else:
    #             table_unit_str_list.append(self._unparse(table_id_or_sql))
    #     cond_str_list = self._unparse_condition(conds, return_list=True)
    #     assert all(x != 'or' for x in cond_str_list)
    #     cond_str_list = [x for x in cond_str_list if x not in ('and', 'or')]
    #     # assert len(table_unit_str_list) == len(cond_str_list) + 1  # assertion on number of join condition
    #     str_segs = [table_unit_str_list[0]]
    #     if include_join:
    #         for i in range(1, len(table_unit_str_list)):
    #             str_segs.append('JOIN')
    #             str_segs.append(table_unit_str_list[i])
    #         if cond_str_list:
    #             str_segs.append('ON')
    #             str_segs.append(cond_str_list[0])
    #             for i in range(1, len(cond_str_list)):
    #                 str_segs.append('AND')
    #                 str_segs.append(cond_str_list[i])
    #     return 'FROM ' + ' '.join(str_segs)

    def _unparse_where(self, _where):
        condition = _where
        # get all selected tables.
        # in selected table: table.column op val
        # else: @ op val
        def get_from_tables(from_dict):
            ret = []
            for from_item in from_dict['table_units']:
                if from_item[0] == 'sql':
                    ret += get_from_tables(from_item[1])
                else:
                    ret.append(from_item[1])
            return list(set(ret))
        sel_table_list = get_from_tables(self.current_sql_dict['from'])
        cond_str_list = []
        cond_table_list = []
        for cond_unit in condition:
            if cond_unit in ('and', 'or'):
                cond_str_list.append(cond_unit)
            else:
                # cond unit
                not_op, op_id, val_unit, val1, val2 = cond_unit
                op_str = WHERE_OPS[op_id]
                # val_unit
                unit_op, col_unit1, col_unit2 = val_unit
                col1_table_id = self.current_table['column_names_original'][col_unit1[1]][0]
                cond_table_list.append(col1_table_id)
                col_unit_str = self._unparse_col_unit(col_unit1, default_table_id=sel_table_list[0])
                if unit_op != 0:
                    print('Warning: calculation between columns are used')
                    try:
                        assert col_unit1[1] == col_unit2[1], "column calculation exception"
                    except:
                        pass
                    col2_table_id = self.current_table['column_names_original'][col_unit2[1]][0]
                    cond_table_list.append(col2_table_id)
                    unit_op_str = UNIT_OPS[unit_op]
                    col_unit2_str = self._unparse_col_unit(col_unit2, default_table_id=sel_table_list[0])
                    col_unit_str = ' '.join([col_unit_str, unit_op_str, col_unit2_str])
                val1_str = self._unparse_val(val1, short_version=True)
                val2_str = self._unparse_val(val2, short_version=True)
                if not_op:
                    assert op_str.lower() in ('in', 'like'), f"{op_str} found"  # todo: check here
                    op_str = 'NOT ' + op_str
                # if 'id' in col_unit_str.lower():  # todo: use this hard rule to filter unimplicit condition?
                #     col_unit_str = '@'
                if 'between' not in op_str.lower():
                    cond_str_list.append(f'{col_unit_str} {op_str} {val1_str}')
                else:
                    assert op_str.lower() == 'between'
                    cond_str_list.append(f'{col_unit_str} {op_str} {val1_str} AND {val2_str}')
        new_cond_table_list = [x for x in cond_table_list if x not in sel_table_list]
        if not cond_str_list:
            where_clause = ''
        else:
            where_clause = ' '.join(cond_str_list)
        if new_cond_table_list:
            join_clause = 'JOIN ' + self.current_table['table_names_original'][new_cond_table_list[-1]] + ' .*'
        else:
            join_clause = ''
        return where_clause, join_clause

    # def _unparse_groupby(self, _groupby):
    #     gb_str_list = []
    #     for gb_item in _groupby:
    #         gb_str = self._unparse_col_unit(gb_item)
    #         gb_str_list.append(gb_str)
    #     return 'GROUP BY ' + ', '.join(gb_str_list)

    def _unparse_orderby(self, _orderby):
        order_op_str = _orderby[0].upper()
        val_unit_str_list = []
        # todo: can default_table_id be a sql? should use the first one?
        default_table_id = self.current_sql_dict['from']['table_units'][0][1]
        for val_unit in _orderby[1]:
            unit_op, col_unit1, col_unit2 = val_unit
            col_unit_str = self._unparse_col_unit(col_unit1, default_table_id)
            if unit_op != 0:
                print('Warning: calculation between columns are used')
                col_unit2_str = self._unparse_col_unit(col_unit2, default_table_id)
                col_unit_str = ' '.join([col_unit_str, UNIT_OPS[unit_op], col_unit2_str])
            val_unit_str_list.append(col_unit_str)
        clause = 'ORDER BY ' + ', '.join(val_unit_str_list) + ' ' + order_op_str
        return clause

    def _unparse_having(self, _having):
        if self.current_sql_dict['groupBy']:
            # todo: is group table unique?
            group_by_column = self.current_sql_dict['groupBy'][0][1]
            default_table_id, column_name = self.current_table['column_names_original'][group_by_column]
        else:
            # todo: can default_table_id be a sql? should use the first one?
            default_table_id = self.current_sql_dict['from']['table_units'][0][1]
        clause = self._unparse_condition(_having, default_table_id=default_table_id)
        return clause

    def _unparse_limit(self, limit):
        return 'LIMIT ' + str(limit)

    def _unparse_col_unit(self, col_unit, default_table_id=None):
        agg_id, col_id, is_distinct = col_unit
        clause = ''
        table_id, column_name = self.current_table['column_names_original'][col_id]
        if table_id >= 0:
            column_name = self.current_table['table_names_original'][table_id] + '.' + column_name
        else:
            assert default_table_id is not None  # column='*'
            column_name = self.current_table['table_names_original'][default_table_id] + '.' + column_name
        clause += column_name.lower() if self.lower_column else column_name
        if agg_id > 0:
            clause = AGG_OPS[agg_id] + ' ( ' + clause + ' ) '
        if is_distinct:
            clause = 'DISTINCT ' + clause
        return clause

    def _unparse_condition(self, condition, default_table_id=None, return_list=False):
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
                col_unit_str = self._unparse_col_unit(col_unit1, default_table_id)
                if unit_op != 0:
                    print('Warning: calculation between columns are used')
                    unit_op_str = UNIT_OPS[unit_op]
                    col_unit2_str = self._unparse_col_unit(col_unit2, default_table_id)
                    col_unit_str = ' '.join([col_unit_str, unit_op_str, col_unit2_str])
                val1_str = self._unparse_val(val1, short_version=True)
                val2_str = self._unparse_val(val2, short_version=True)
                if not_op:
                    assert op_str.lower() in ('in', 'like'), f"{op_str} found"  # todo: check here
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

    def _unparse_val(self, val, short_version=False):
        if val is None:
            return None
        if isinstance(val, str):
            val_str = val
        elif isinstance(val, dict):
            val_str = self._unparse(val)
        elif isinstance(val, int) or isinstance(val, float):
            val_str = str(int(val))
        else:
            val_str = self._unparse_col_unit(val)
        return val_str


if __name__ == '__main__':
    parser = SqlDictParser('data/cosql/tables.json')
    # dev_data = json.load(open('data/cosql/train.json'))
    # for i in range(len(dev_data)):
    #     print(i)
    #     for j in range(len(dev_data[i]['interaction'])):
    #         case_db = dev_data[i]['database_id']
    #         case = dev_data[i]['interaction'][j]['sql']
    #         sql = dev_data[i]['interaction'][j]['query']
    #         print(sql)
    #         print(parser.unparse(case_db, case))
    #         print

    train_data = json.load(open('data/spider/train_spider.json'))
    for i in range(99, 200):
        print(i)
        db_id = train_data[i]['db_id']
        sql = train_data[i]['query']
        sql_dict = train_data[i]['sql']
        print(sql)
        natsql = parser.unparse(i, db_id, sql_dict)
        print(natsql)


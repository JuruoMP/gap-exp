from seq2seq.lf_util.natsql.utils import *
import copy

SQL_TOP = 1  # top sql, based on 'except', 'intersect', 'union'
SQL_SUB = 2  # sub sql,
SQL_SUBSUB = 3  # sub-sub sql, based on 'sub'


# NatSQL V1.1:
# add group by
# support where + IUE
# support move order by to subquery automatically
# extend join on: table_1.*=table_2.* -> join these tables. If table_1 == table_2, only join one more table.

# NatSQL V1.2:
# add un-foreign key search
# add @.@ column search for subquery
# add group by control for exact match

# NatSQL V1.3:
# separate the Join on
# add new key-word join

class Args():
    def __init__(self):
        self.in_sub_sql = True  # if True: whenever see "in col",it will be transferred to "in (select col)",else: the left and right col are foreign key relationship will be tranferred to "join on"
        self.eq_sub_sql = False  # if True: whenever see "= col", it will be transferred to "= (select col)", else: only left and right col are not foreign key relationship will be tranferred
        self.keep_top_order_by = True  # We previous plan to move "order by" to where condition. But now it is not used. So please keep it to be True.
        self.orderby_to_subquery = True  # Move the order by to the subquery when there is subquery and this order by should be in the subquery.
        self.group_for_exact_match = True  # For get a higher score in exact match in spider benchmark


class table_With_FK:
    def __init__(self, table, keys):
        self.table = table
        self.keys = keys

    def try_get_join_table(self, table_list, column):
        table_ = column[self.keys[0]][1].split('.')[0]
        if (table_.lower() in table_list and self.table.lower() not in table_list):
            return self.table
        if (table_.lower() not in table_list and self.table.lower() in table_list):
            return table_
        return None

    def return_join_on(self, table, column):
        # if self.table not in table_list:
        #     sql = " join " + self.table
        # else:
        #     sql = " join " + column[self.keys[0]][1].split('.')[0]
        sql = " join " + table
        sql += " on " + column[self.keys[0]][1] + " = " + column[self.keys[1]][1]
        return sql


globe_join_on_label_count = 0


def check_relation(foreign_keys, primary_keys, column_names, left_column, right_column, left_tables=None):
    try:
        (k_1, k_2) = -1, -1
        if left_tables and not left_column:  # means the left_column is @.@, so input set it to be None
            left_col = None
        else:
            left_col = val_unit_back(left_column).lower()
        right_col = col_unit_back(right_column).lower()
        for idx, cn in enumerate(column_names):
            if left_col == cn[1].lower():
                k_1 = idx
            if right_col == cn[1].lower():
                k_2 = idx
        if right_col == "*" and k_2 == -1 and k_1 >= 0:
            right_col = right_column[1].split(".")[0].lower()
            for fk in foreign_keys:
                if fk[0] == k_1 and column_names[fk[1]][1].split('.')[0].lower() == right_col:
                    k_2 = fk[1]
                elif fk[1] == k_1 and column_names[fk[0]][1].split('.')[0].lower() == right_col:
                    k_2 = fk[0]
        elif not left_col and left_tables and k_1 == -1 and k_2 >= 0:
            for fk in foreign_keys:
                if fk[0] == k_2 and column_names[fk[1]][1].split('.')[0].lower() in left_tables:
                    k_1 = fk[1]
                elif fk[1] == k_2 and column_names[fk[0]][1].split('.')[0].lower() in left_tables:
                    k_1 = fk[0]
        elif right_col == "*" and left_col == "*" and k_2 == -1 and k_1 == -1:
            right_col = right_column[1].split(".")[0].lower()
            left_col = left_column[1][1].split(".")[0].lower()
            for fk in foreign_keys:
                if column_names[fk[0]][1].split('.')[0].lower() == right_col and column_names[fk[1]][1].split('.')[
                    0].lower() == left_col:
                    return fk
                if column_names[fk[0]][1].split('.')[0].lower() == left_col and column_names[fk[1]][1].split('.')[
                    0].lower() == right_col:
                    return fk
        if [k_1, k_2] in foreign_keys or [k_2, k_1] in foreign_keys:
            return [k_1, k_2]
        elif k_1 >= 0 and k_2 >= 0 and (k_1 in primary_keys or k_2 in primary_keys) and column_names[k_1][0] != \
                column_names[k_2][0] and left_column[0] == 0 and left_column[1][0] == 0 and right_column[0] == 0:
            return [k_1, k_2]
    except:
        pass
    return None


def condition_str(column):
    if WHERE_OPS[column[1]] == 'between':
        return val_unit_back(column[2]) + " between " + str(column[3]) + ' and ' + str(column[4]) + " "
    elif type(column[3]) == list:
        return val_unit_back(column[2]) + " " + WHERE_OPS[column[1]] + " " + column[3][1] + " "
    else:
        return val_unit_back(column[2]) + " " + WHERE_OPS[column[1]] + " " + str(column[3]) + " "


def get_where_column(sql_dict, table_list, start_index, sql_type, table_json, args):
    AND_OR_TYPE = 1
    SUB_SQL_TYPE = 2
    SUB_SUB_SQL_TYPE = 3
    TOP_SQL_TYPE = AND_OR_TYPE

    break_idx = -1
    next_type = None
    last_column_type = None
    see_sub_sql = False

    next_table_list = []
    sql_str = " "
    having = " "
    order_by = ''

    for idx, column in enumerate(sql_dict['where']):
        if idx < start_index:
            continue

        if isinstance(column, str) and column.lower() in SPECIAL_COND_OPS:  # 'except', 'intersect', 'union', 'sub'
            if column == 'sub':
                last_column_type = SUB_SQL_TYPE
                break_idx, next_type = (idx + 1, SQL_SUBSUB) if (idx > start_index and not next_type) or (
                            sql_type == SQL_SUB and not next_type) else (break_idx, next_type)
                # next_type = SQL_SUBSUB if idx > start_index and not next_type else next_type
                if sql_type == SQL_SUBSUB:
                    break
            else:
                break_idx, next_type = (idx, SQL_TOP) if (idx > start_index or break_idx == -1) and not next_type else (
                break_idx, next_type)
                break
        elif isinstance(column, str) and column.lower() in AND_OR_OPS:  # 'and', 'or'
            last_column_type = AND_OR_TYPE
            if sql_str.endswith('and '):
                sql_str = sql_str[:-4]
            elif sql_str.endswith('or '):
                sql_str = sql_str.strip()[:-3]
            if sql_str != " ":
                sql_str += column + " "
            continue
        else:
            assert not column[0]
            if type(column[3]) == list:
                if column[2][1][1] == "@.@":
                    fk = check_relation(table_json['foreign_keys'], table_json['primary_keys'],
                                        table_json['table_column_names_original'], None, column[3], table_list)
                else:
                    fk = check_relation(table_json['foreign_keys'], table_json['primary_keys'],
                                        table_json['table_column_names_original'], column[2], column[3])
                (table_left, column_left) = (c.lower() for c in column[2][1][1].split('.'))
                (table_right, column_right) = (c.lower() for c in column[3][1].split('.'))
                if column[1] == 15:  # jion:
                    if break_idx != -1 or next_type:
                        continue
                    table_list.append(table_right)
                elif ((not args.in_sub_sql and column[1] == 8) or (not args.eq_sub_sql and column[1] == 2)) and fk:
                    if break_idx != -1 or next_type:
                        continue
                    if table_left not in table_list and table_left != "@":  # V1.2
                        table_list.append(table_left)
                    table = table_right  # column[3][1].split('.')[0]
                    twfk = table_With_FK(table, fk)
                    table_list.append(twfk)
                elif table_left == table_right and column_left != column_right and column[1] >= 2 and column[
                    1] <= 7 and not fk and (not (column[2][0] or column[2][1][0] or column[3][0])):
                    # different column in the same table
                    if break_idx != -1 or next_type or see_sub_sql:
                        continue
                    sql_str += condition_str(column)  # for columnA != columnB in the same table
                elif column[2][0] or column[2][1][0] and column[3][0] in [1, 2]:
                    # condition to order by
                    if break_idx != -1 or next_type or see_sub_sql:
                        continue
                    order_by = " ORDER BY " + val_unit_back(column[2]) + (
                        " DESC LIMIT 1 " if column[3][0] == 1 else " ASC LIMIT 1 ")
                # elif "*" in column_right and "*" in column_left: #V1.1
                elif "*" in column_right and column[1] == 2 and column[3][0] == 0:  # V1.2
                    # ? = * to join on
                    if break_idx != -1 or next_type or see_sub_sql:
                        continue
                    if table_left not in table_list and table_left != "@":  # V1.2
                        table_list.append(table_left)
                    if fk:
                        twfk = table_With_FK(table_right, fk)
                        table_list.append(twfk)
                    elif table_right not in table_list:
                        table_list.append(table_right)
                # elif "*" in column_right and "*" in column_left: # table.* opt table.*
                #     if table_left not in table_list and table_left != "@": #V1.2
                #         table_list.append(table_left)
                #     # if column[1] == 2 and fk: # =, so try to jion on:
                #     #     twfk = table_With_FK(table_right,fk)
                #     #     table_list.append(twfk)
                #     # else: # sub query
                #     break_idx,next_type = (idx,SQL_SUB) if (idx > start_index and not next_type) or (sql_type == SQL_TOP and not next_type) else (break_idx,next_type)
                #     column = infer_at_col(column,table_list,table_json)
                #     table_left = column[2][1][1].split(".")[0].lower()
                #     # if column[3][1].split(".")[0].lower() not in next_table_list:#!= table_right.lower():
                #     #     next_table_list.append(column[3][1].split(".")[0].lower())
                #     sql_str += column[2][1][1] + " " + WHERE_OPS[column[1]] + " (@@@) "
                else:
                    order_by_ = maybe_order_by(sql_dict, table_list, idx, start_index, next_type, sql_type, table_json,
                                               args)
                    order_by += order_by_
                    if not order_by_:
                        break_idx, next_type = (idx, SQL_SUB) if (idx > start_index and not next_type) or (
                                    sql_type == SQL_TOP and not next_type) else (break_idx, next_type)
                        if last_column_type and sql_type > last_column_type:
                            # (and + list) sql_type = 1 = new sub sql, it end 2 and 3(SQL_SUB, SQL_SUBSUB); sub + list = 2 = new subsub sql, it end 3(SQL_SUBSUB)
                            break
                        if last_column_type and sql_type != last_column_type:  # it will true, when last is sub, and sql_type is
                            # Filter by last if: 1.SQL_SUB meet 'and list sub sql model' 2. SQL_SUBSUB meet 'and list sub sql model' 3. SQL_SUBSUB meet 'sub list subsub sql model'
                            # It should True here when: 1. SQL_TOP meet 'sub list subsub sql model'
                            continue
                        # It should run here when: 1. SQL_TOP meet 'and list sub sql model'; 2. SQL_SUB meet 'sub list sub sql model'
                        see_sub_sql = True
                        if column[2][1][1] == "@.@":  # V1.2
                            column = infer_at_col(column, table_list, table_json)
                            table_left = column[2][1][1].split(".")[0].lower()
                            # if column[3][1].split(".")[0].lower() not in next_table_list:#!= table_right.lower():
                            #     next_table_list.append(column[3][1].split(".")[0].lower())
                        elif ".*" in column[3][1] and fk:
                            column[3][1] = table_json['table_column_names_original'][fk[1]][1] if column[2][1][
                                                                                                      1].lower() == \
                                                                                                  table_json[
                                                                                                      'table_column_names_original'][
                                                                                                      fk[0]][
                                                                                                      1].lower() else \
                            table_json['table_column_names_original'][fk[0]][1]
                        if column[2][0] or column[2][1][0]:
                            having += val_unit_back(column[2]).lower() + " " + WHERE_OPS[column[1]] + " (@@@) "
                        else:
                            if sql_str.strip() and not sql_str.strip().endswith("and") and not sql_str.strip().endswith(
                                    "or"):
                                sql_str += " and "
                            sql_str += column[2][1][1].lower() + " " + WHERE_OPS[column[1]] + " (@@@) "
                table = table_left.lower()  # column[2][1][1].split('.')[0]
            elif not see_sub_sql:
                table = column[2][1][1].split('.')[0].lower()
                if column[2][1][1] == "@.@" and idx >= 2:  # V1.2
                    column[2][1][1] = sql_dict['where'][idx - 2][2][1][1]
                if column[2][0] or column[2][1][0]:  # having
                    and_or = sql_dict['where'][idx - 1] + " " if idx > 0 and type(
                        sql_dict['where'][idx - 1]) == str else None
                    if and_or and having != " ":
                        having += and_or
                    having += condition_str(column)
                    if and_or and sql_str.endswith(and_or):
                        sql_str = sql_str[:-len(and_or)]
                    pass
                else:  # where
                    sql_str += condition_str(column)

            if table and table not in table_list and table != "@":  # V1.2
                table_list.append(table)

    sql_str = sql_str if sql_str == " " else " where " + sql_str
    while True:
        if sql_str.strip().endswith('and'):
            sql_str = " " + sql_str.strip()[:-3] + " "
        elif sql_str.strip().endswith('or'):
            sql_str = " " + sql_str.strip()[:-2] + " "
        else:
            break

    having = having if having == " " else " having " + having

    return break_idx, table_list, next_type, sql_str, having, order_by, next_table_list


def get_table_network(table_json, table_list, join_on_label, re_single=True):
    def create_foreign_key(table_json, table_idx_list, restrict=True):
        """
        len(table_idx_list) must be two!!!
        """

        def restrict_check(table_json, pkey):
            if table_json['column_types'][pkey[0][0]] == table_json['column_types'][pkey[1][0]]:
                ws = table_json['column_names'][pkey[0][0]][1].split(" ")
                for w in ws:
                    if w in table_json['column_names'][pkey[1][0]][1]:
                        return [[[pkey[0][0], pkey[1][0]]], [pkey[0][1], pkey[1][1]]]
            return None

        def super_key_create_2_to_3(table_json, pkey):
            # try to find a new table as bridge to connect this two table
            for ii in range(2):
                pk_l = pkey[0][0] if ii == 0 else pkey[1][0]
                pk_r = pkey[1][0] if ii == 0 else pkey[0][0]
                table_l = table_json['column_names'][pk_l][0]
                table_r = table_json['column_names'][pk_r][0]
                for i, fk in enumerate(table_json['foreign_keys']):
                    if pk_l in fk:
                        other_fk = fk[1] if pk_l == fk[0] else fk[0]
                        # other_tble = table_json['column_names'][other_fk][0]
                        for j, fk2 in enumerate(table_json['foreign_keys']):
                            if i == j:
                                continue
                            if other_fk in fk2:  # jump here:
                                jump_fk = fk2[1] if other_fk == fk2[0] else fk2[0]
                                jump_tble = table_json['column_names'][jump_fk][0]
                                for k, fk3 in enumerate(table_json['foreign_keys']):
                                    if k in (i, j):
                                        continue
                                    if pk_r in fk3:
                                        jump_fk2 = fk3[1] if pk_r == fk3[0] else fk3[0]
                                        if table_json['column_names'][jump_fk2][0] == jump_tble:
                                            return [[[pk_l, jump_fk], fk3], [table_l, jump_tble, table_r]]
            return None

        pkey = []  # try to directly use the primary key to join
        for k in table_json['primary_keys']:
            if table_json['column_names'][k][0] in table_idx_list:
                pkey.append([k, table_json['column_names'][k][0]])
        potential_fk = []
        for pk in pkey:  # same name with primary key to become JOIN ON
            other_table = table_idx_list[1] if pk[1] == table_idx_list[0] else table_idx_list[0]
            for i, o_col, col in zip(range(len(table_json['column_names_original'])),
                                     table_json['column_names_original'], table_json['column_names']):
                if col[0] == other_table and (
                        table_json['column_names_original'][pk[0]][1] == o_col[1] or table_json['column_names'][pk[0]][
                    1] == col[1]) and table_json['column_types'][pk[0]] == table_json['column_types'][i]:
                    potential_fk.append([[[pk[0], i]], [pk[1], other_table]])
        if potential_fk:
            return potential_fk[0]
        else:
            if len(pkey) == 2:
                if restrict:
                    r_ = restrict_check(table_json, pkey)
                    if r_:
                        return r_
                else:
                    tale_net = super_key_create_2_to_3(table_json, pkey)
                    if tale_net:
                        return tale_net
                    return [[[pkey[0][0], pkey[1][0]]], [pkey[0][1], pkey[1][1]]]
            # try to directly use the same column name key to join
            for i, c_n, c_on in zip(range(len(table_json['column_names'])), table_json['column_names'],
                                    table_json['column_names_original']):
                if c_n[0] == table_idx_list[0]:
                    for j, c_n2, c_on2 in zip(range(len(table_json['column_names'])), table_json['column_names'],
                                              table_json['column_names_original']):
                        if c_n2[0] == table_idx_list[1]:
                            if (c_n[1] == c_n2[1] or c_on[1] == c_on2[1]) and c_on[1] not in ["id", "name", "ids", "*"]:
                                return [[[i, j]], table_idx_list]

            # try to directly use the foreign key to join
            r_ = None
            pkey = []
            tables = []
            for k in table_json['foreign_keys']:
                if table_json['column_names'][k[0]][0] in table_idx_list and table_json['column_names'][k[0]][
                    0] not in tables:
                    pkey.append([k[0], table_json['column_names'][k[0]][0]])
                    tables.append(table_json['column_names'][k[0]][0])
                if table_json['column_names'][k[1]][0] in table_idx_list and table_json['column_names'][k[1]][
                    0] not in tables:
                    pkey.append([k[1], table_json['column_names'][k[1]][0]])
                    tables.append(table_json['column_names'][k[1]][0])
            if len(pkey) == 2:
                if restrict:
                    r_ = restrict_check(table_json, pkey)
                else:
                    return [[[pkey[0][0], pkey[1][0]]], [pkey[0][1], pkey[1][1]]]

        return r_

    def get_fk_network(table_json, table_list, return_num=1):
        table_index_list = []  # change table name list to a number list
        table_fk_list = []

        for t in table_list:
            if isinstance(t, table_With_FK):
                table_fk_list.append(t)
            else:
                if t.lower() != '@':  # V1.2
                    table_index_list.append([n.lower() for n in table_json['table_names_original']].index(
                        t.lower()))  # (table_json['table_names_original'].index(t))

        # min_network = 9999
        # from_table_net = []
        # for idx, network in enumerate(table_json['network']):
        #     len_net = len(network[1])
        #     if len_net >= len(table_index_list) and len_net < min_network:
        #         success = True
        #         for t in table_index_list:
        #             if t not in network[1]:
        #                 success = False
        #                 break
        #         if success:
        #             min_network = len_net
        #             from_table_net = copy.deepcopy(network)
        from_table_net = []
        for idx, network in enumerate(table_json['network']):
            if len(network[1]) == len(table_index_list) or (
                    len(network[1]) > len(table_index_list) and len(table_index_list) > 1):
                success = True
                for t in table_index_list:
                    if t not in network[1]:
                        success = False
                        break
                if success and network[1][0] in table_index_list and network[1][-1] in table_index_list:
                    from_table_net.append(copy.deepcopy(network))
        if not from_table_net:
            for idx, network in enumerate(table_json['network']):
                if len(network[1]) == len(table_index_list) or (
                        len(network[1]) > len(table_index_list) and len(table_index_list) > 1):
                    success = True
                    for t in table_index_list:
                        if t not in network[1]:
                            success = False
                            break
                    if success:
                        from_table_net.append(copy.deepcopy(network))
        # re-order
        if from_table_net:
            from_table_net = sorted(from_table_net, key=lambda x: len(x[1]))

        if from_table_net and table_fk_list:
            for net in from_table_net:
                for tf in table_fk_list:
                    if (table_json['column_names'][tf.keys[0]][0] in net[1] and table_json['column_names'][tf.keys[1]][
                        0] not in net[1]) or (table_json['column_names'][tf.keys[1]][0] in net[1] and
                                              table_json['column_names'][tf.keys[0]][0] not in net[1]):
                        net[0].append(tf.keys)
                        net[1].append([n.lower() for n in table_json['table_names_original']].index(tf.table.lower()))
        if return_num and from_table_net:
            return from_table_net[0], table_fk_list, table_index_list
        else:
            return from_table_net, table_fk_list, table_index_list

            # c = 0

    # for i in table_list:
    #     if isinstance(i,str):
    #         c += 1
    # if c > 2:
    #     print("^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    from_table_net, table_fk_list, table_index_list = get_fk_network(table_json, table_list, 0)

    if not from_table_net and len(table_index_list) == 2:
        if re_single:
            return create_foreign_key(table_json, table_index_list, False), table_fk_list
        else:
            return [create_foreign_key(table_json, table_index_list, False)], table_fk_list
    elif not from_table_net and len(table_index_list) == 3:
        from_table_net1, table_fk_list1, table_index_list1 = get_fk_network(table_json, [table_list[0], table_list[1]])
        from_table_net2, table_fk_list2, table_index_list2 = get_fk_network(table_json, [table_list[0], table_list[2]])
        from_table_net3, table_fk_list3, table_index_list3 = get_fk_network(table_json, [table_list[1], table_list[2]])
        net1 = None
        net2 = None
        net3 = None
        if not from_table_net1 and not from_table_net2 and from_table_net3:
            net1 = create_foreign_key(table_json, [table_index_list[0], table_index_list[1]])
            net2 = create_foreign_key(table_json, [table_index_list[0], table_index_list[2]])
            net3 = copy.deepcopy(from_table_net3)
        elif not from_table_net1 and from_table_net2 and not from_table_net3:
            net1 = create_foreign_key(table_json, [table_index_list[0], table_index_list[1]])
            net2 = create_foreign_key(table_json, [table_index_list[2], table_index_list[1]])
            net3 = copy.deepcopy(from_table_net2)
        elif from_table_net1 and not from_table_net2 and not from_table_net3:
            net1 = create_foreign_key(table_json, [table_index_list[1], table_index_list[2]])
            net2 = create_foreign_key(table_json, [table_index_list[0], table_index_list[2]])
            net3 = copy.deepcopy(from_table_net1)
        elif from_table_net1 and from_table_net2 and not from_table_net3:
            net1 = copy.deepcopy(from_table_net1)
            net3 = copy.deepcopy(from_table_net2)
        elif from_table_net1 and not from_table_net2 and from_table_net3:
            net1 = copy.deepcopy(from_table_net1)
            net3 = copy.deepcopy(from_table_net3)
        elif not from_table_net1 and from_table_net2 and from_table_net3:
            net1 = copy.deepcopy(from_table_net3)
            net3 = copy.deepcopy(from_table_net2)
        if net1 and not net2:
            # combine the net1
            net3[0].append(net1[0][0])
            net3[1].append(net1[1][0] if net1[1][0] not in net3[1] else net1[1][1])
        elif not net1 and net2:
            net3[0].append(net2[0][0])
            net3[1].append(net2[1][0] if net2[1][0] not in net3[1] else net2[1][1])
        elif net1 and net2:
            if net1[0][0][0] in table_json["primary_keys"] or net1[0][0][1] in table_json["primary_keys"]:
                net2 = net1
            net3[0].append(net2[0][0])
            net3[1].append(net2[1][0] if net2[1][0] not in net3[1] else net2[1][1])

        if len(net3[1]) == 3:
            if re_single:
                return net3, table_fk_list
            else:
                return [net3], table_fk_list
    if from_table_net and re_single:
        if join_on_label:
            global globe_join_on_label_count
            idx = join_on_label[globe_join_on_label_count]
            globe_join_on_label_count += 1
        else:
            idx = 0
        return from_table_net[idx], table_fk_list
    return from_table_net, table_fk_list


def create_from_table(from_table_net, table_names, column, table_fk_list):
    if not from_table_net:
        # print("Can not find correct table network")
        pass
    assert from_table_net, "Can not find correct table network"
    table_use = []
    table_fk_remain = []
    sql = None
    if not from_table_net[0]:  # only one table
        sql = " from " + table_names[from_table_net[1][0]]
        table_use.append(table_names[from_table_net[1][0]].lower())

    table_fk_remain = copy.deepcopy(table_fk_list)

    for fk in from_table_net[0]:
        if not sql:
            sql = " from " + table_names[column[fk[0]][0]]
            table_use.append(table_names[column[fk[0]][0]].lower())

        table_fk_list = []
        for tfk in table_fk_remain:  # it is impossible to replace one network when start table is not table_names[from_table_net[1][0]] in this code. it can be update later.
            join_table = tfk.try_get_join_table(table_use, column)
            if join_table:
                sql += tfk.return_join_on(join_table, column)
                table_use.append(join_table.lower())
            else:
                table_fk_list.append(tfk)
        table_fk_remain = copy.deepcopy(table_fk_list)

        if table_names[column[fk[1]][0]].lower() not in table_use:
            sql += " join " + table_names[column[fk[1]][0]]
            sql += " on " + column[fk[0]][1] + " = " + column[fk[1]][1]
            table_use.append(table_names[column[fk[1]][0]].lower())
        elif table_names[column[fk[0]][0]].lower() not in table_use:
            sql += " join " + table_names[column[fk[0]][0]]
            sql += " on " + column[fk[1]][1] + " = " + column[fk[0]][1]
            table_use.append(table_names[column[fk[0]][0]].lower())

    for tfk in table_fk_remain:
        join_table = tfk.try_get_join_table(table_use, column)
        if join_table:
            sql += tfk.return_join_on(join_table, column)
            table_use.append(join_table.lower())

    return sql


def create_order_by(order_dict, limit):
    orderby = ""
    table_list = []
    agg_in_order_bool = False
    if order_dict:
        orderby = " order by "
        orderby += ",".join([val_unit_back(order) for order in order_dict[1]])
        orderby += " " + order_dict[0]

        agg_in_order = [val_unit_contain_agg(order) for order in order_dict[1]]
        agg_in_order_bool = True if True in agg_in_order else False

        for order in order_dict[1]:
            table = order[1][1].split('.')[0].lower()
            if table not in table_list:
                table_list.append(table)
    if limit:
        orderby += " limit " + str(limit) + " "
    return orderby, table_list, agg_in_order_bool


def maybe_order_by(sql_dict, table_list, idx, start_index, next_sql_type, sql_type_now, table_json, args):
    """
     order by and sub sql ->
     order by -> return order by
     and sub-sql and order-by
     and sub-sql sub subsub-sql

    """

    if (idx == start_index and sql_type_now != SQL_TOP) or (
            next_sql_type == SQL_SUBSUB and idx > 0 and sql_dict['where'][idx - 1] == 'sub'):
        return ''

    # print(len(sql_dict['where']))
    if (sql_dict['where'][idx][3][0] == 1 or sql_dict['where'][idx][3][0] == 2) and sql_dict['where'][idx][2][1][1] == \
            sql_dict['where'][idx][3][1] and sql_dict['where'][idx][1] == 2:  # sql_dict['where'][idx][1] == 2 means '='
        break_idx, next_type = (idx, SQL_SUB) if (idx > start_index and not next_sql_type) or (
                    sql_type_now == SQL_TOP and not next_sql_type) else (idx, next_sql_type)
        break_idx, table_list, next_sql, sql_where, sql_having, orderby_sql_, next_table_list = get_where_column(
            sql_dict, table_list, break_idx + 1, next_type, table_json, args)
        if (not next_sql and len(sql_dict['where']) == idx + 1) or (
                next_sql and break_idx <= idx + 2):  # the end of the sql or sub sql.
            if next_sql_type:  # make it pass for next if check in get_where_column
                return ' '

            if args.keep_top_order_by and sql_type_now == SQL_TOP:
                return ''

            order_by_sql = " ORDER BY " + sql_dict['where'][idx][2][1][1] + (
                " DESC LIMIT 1 " if sql_dict['where'][idx][3][0] == 1 else " ASC LIMIT 1 ")
            return order_by_sql
    return ''


def is_there_subquery(wheres):
    """
        V1.1
    """
    if not wheres:
        return False
    for where in wheres:
        if isinstance(where, list) and isinstance(where[3], list):
            if where[2][1][1] == "@.@" and "*" in where[3][1] and where[1] == 2:
                continue
            return True
    return False


def is_orderby_for_subquery(sql_dict):
    """
        V1.1
    """
    if not sql_dict['limit'] or not sql_dict['orderBy']:
        return False
    if is_there_subquery(sql_dict['where']):
        return True
    return False


def orderby_to_subquery(sql_dict, tb_list):
    """
        V1.1
    """
    orderby_sql = ""
    if is_orderby_for_subquery(sql_dict):
        orderby_sql, table_list, agg_in_order = create_order_by(sql_dict['orderBy'], sql_dict['limit'])
        for t in table_list:
            if t not in tb_list:
                tb_list.append(t)
    return orderby_sql, tb_list


def primary_keys(table_json, table_id):
    for key in table_json['primary_keys']:
        if table_json['column_names'][key][0] == table_id:
            return key
    for key, col in enumerate(table_json['column_names']):
        if table_json['column_names'][key][0] == table_id:
            col_n = table_json['column_names'][key][1].lower().strip()
            col_n = col_n.replace(table_json['table_names'][table_id].lower(), "").strip()
            col_n = col_n.replace(table_json['table_names_original'][table_id].lower(), "").strip()
            if col_n in ["id", "ids"]:
                return key
    return -1


def infer_at_col(column, table_list, table_json):
    """V1.2
    infer the @.@ column value
    """

    def find_fk_cols(l_table, r_table, table_json, r_col=None):
        if r_col:
            for fk in table_json['foreign_keys']:
                if r_col == fk[0] and table_json['column_names'][fk[1]][0] == l_table:
                    return True, fk[1], r_col
                elif r_col == fk[1] and table_json['column_names'][fk[0]][0] == l_table:
                    return True, fk[0], r_col
        else:
            for fk in table_json['foreign_keys']:
                if table_json['column_names'][fk[0]][0] == r_table and table_json['column_names'][fk[1]][0] == l_table:
                    return True, fk[1], fk[0]
                elif table_json['column_names'][fk[1]][0] == r_table and table_json['column_names'][fk[0]][
                    0] == l_table:
                    return True, fk[0], fk[1]
        return False, 0, 0

    def find_col(l_tables, r_table, r_col, table_json):
        # foreign key search:
        r_col_idx = 0
        if r_col != "*":
            for i, col in enumerate(table_json['column_names_original']):
                if col[0] == r_table and r_col.lower() == col[1].lower():
                    r_col_idx = i
                    break

        for lt in l_tables[1]:
            sucess, left, right = find_fk_cols(lt, r_table, table_json, r_col_idx)
            if sucess:
                return left, right

        # same name key search:
        if r_col != "*":
            for lt in l_tables[1]:
                for i, col_o, col in zip(range(len(table_json['column_names_original'])),
                                         table_json['column_names_original'], table_json['column_names']):
                    if col_o[0] == lt and (col_o[1] == r_col or col[1] == table_json['column_names'][r_col_idx][1]):
                        return i, r_col_idx
            if table_json['column_names'][r_col_idx][1].count(" ") == 2:
                # three match two to return
                col_r_names = table_json['column_names'][r_col_idx][1].split(" ")
                for lt in l_tables[1]:
                    for i, col in zip(range(len(table_json['column_names_original'])), table_json['column_names']):
                        if col[0] == lt and (col[1].count(" ") == 1):
                            c_ls = col[1].split(" ")
                            if c_ls[0] in col_r_names and c_ls[1] in col_r_names:
                                return i, r_col_idx
                    for i, col in zip(range(len(table_json['column_names_original'])), table_json['column_names']):
                        if col[0] == lt and (col[1].count(" ") == 0):
                            if col[1] in col_r_names and table_json["table_names"][lt] in col_r_names and col[1] != \
                                    table_json["table_names"][lt]:
                                return i, r_col_idx
            return r_col_idx, r_col_idx
        else:
            result = []
            for j, rcol_o, rcol in zip(range(len(table_json['column_names_original'])),
                                       table_json['column_names_original'], table_json['column_names']):
                if rcol[0] != r_table or rcol_o[1] == '*':
                    continue
                for lt in l_tables[1]:
                    for i, col_o, col in zip(range(len(table_json['column_names_original'])),
                                             table_json['column_names_original'], table_json['column_names']):
                        if col_o[0] == lt and (col_o[1] == rcol_o[1] or col[1] == rcol[1]):
                            result.append([i, j])
            if result:
                for r in result:
                    if table_json['column_names_original'][r[0]][1] not in ["name", "id"] and \
                            table_json['column_names'][r[0]][1] not in ["name", "id"]:
                        return r[0], r[1]
                return result[0][0], result[0][1]
        # if len(l_tables[1]) == 1:
        #     # use the first table:
        #     pl = primary_keys(table_json,l_tables[1][0])
        #     return pl,pl
        if r_table >= 0:
            # use the right table:
            pl = primary_keys(table_json, r_table)
            if pl >= 0:
                return pl, pl
            if len(l_tables[1]) >= 1:
                pl = primary_keys(table_json, l_tables[1][0])
            if pl >= 0:
                return pl, pl
        return 0, 0

    col_right = column[3][1].split(".")
    table_right = col_right[0].lower()
    col_right = col_right[1].lower()
    table_right_idx = [n.lower() for n in table_json['table_names_original']].index(table_right)
    if table_right in table_list:  # It will be the same column for both side
        if col_right == "*":
            primarykey = primary_keys(table_json, table_right_idx)
            if primarykey >= 0:
                column[3][1] = table_json['table_column_names_original'][primarykey][1]
        column[2][1][1] = column[3][1]
    else:
        if len(table_list) > 1:
            from_table_net, table_fk_list = get_table_network(table_json, table_list, None)
            if from_table_net and table_right_idx in from_table_net[1]:
                # There is same tables
                if col_right == "*":
                    primarykey = primary_keys(table_json, table_right_idx)
                    if primarykey >= 0:
                        column[3][1] = table_json['table_column_names_original'][primarykey][1]
                column[2][1][1] = column[3][1]
            elif from_table_net:
                # There isn't same tables, so look for the foreign key relations.
                col_left_idx, col_right_idx = find_col(from_table_net, table_right_idx, col_right, table_json)
                column[3][1] = table_json['table_column_names_original'][col_right_idx][1]
                column[2][1][1] = table_json['table_column_names_original'][col_left_idx][1]
        else:
            # There isn't same tables, so look for the foreign key relations.
            table_left_idx = [n.lower() for n in table_json['table_names_original']].index(table_list[0].lower())
            col_left_idx, col_right_idx = find_col([[], [table_left_idx]], table_right_idx, col_right, table_json)
            column[3][1] = table_json['table_column_names_original'][col_right_idx][1]
            column[2][1][1] = table_json['table_column_names_original'][col_left_idx][1]
    return column


def intersect_check(wheres):
    """V1.2
    infer the intersect
    """
    try:
        if ((len(wheres) == 3 and wheres[1] == "and") or (
                len(wheres) == 5 and wheres[1] == "and" and wheres[3] == "and")) and (
                wheres[0][2][1][1] == wheres[2][2][1][1]):  # or wheres[2][2][1][1] == "@.@"
            if wheres[0][1] == 2 and wheres[2][1] == 2:
                wheres[1] = "intersect_"
            elif wheres[0][1] == 3 and wheres[2][1] == 4 and int(wheres[0][3]) > int(wheres[2][3]):
                wheres[1] = "intersect_"
            elif wheres[0][1] == 4 and wheres[2][1] == 3 and int(wheres[0][3]) < int(wheres[2][3]):
                wheres[1] = "intersect_"
            elif wheres[0][1] == 5 and wheres[2][1] == 6 and int(wheres[0][3]) > int(wheres[2][3]):
                wheres[1] = "intersect_"
            elif wheres[0][1] == 6 and wheres[2][1] == 5 and int(wheres[0][3]) < int(wheres[2][3]):
                wheres[1] = "intersect_"
            elif wheres[0][1] == 4 and wheres[2][1] == 5 and int(wheres[0][3]) < int(wheres[2][3]):
                wheres[1] = "intersect_"
            elif wheres[0][1] == 5 and wheres[2][1] == 4 and int(wheres[0][3]) > int(wheres[2][3]):
                wheres[1] = "intersect_"
            elif wheres[0][1] == 3 and wheres[2][1] == 6 and int(wheres[0][3]) > int(wheres[2][3]):
                wheres[1] = "intersect_"
            elif wheres[0][1] == 6 and wheres[2][1] == 3 and int(wheres[0][3]) < int(wheres[2][3]):
                wheres[1] = "intersect_"
            return wheres
        if (len(wheres) == 5 and wheres[1] == "and" and wheres[3] == "and") and (
                wheres[4][2][1][1] == wheres[2][2][1][1]):
            if wheres[4][1] == 2 and wheres[2][1] == 2:
                wheres[3] = "intersect_"
            elif wheres[4][1] == 3 and wheres[2][1] == 4 and int(wheres[4][3]) > int(wheres[2][3]):
                wheres[3] = "intersect_"
            elif wheres[4][1] == 4 and wheres[2][1] == 3 and int(wheres[4][3]) < int(wheres[2][3]):
                wheres[3] = "intersect_"
            elif wheres[4][1] == 5 and wheres[2][1] == 6 and int(wheres[4][3]) > int(wheres[2][3]):
                wheres[3] = "intersect_"
            elif wheres[4][1] == 6 and wheres[2][1] == 5 and int(wheres[4][3]) < int(wheres[2][3]):
                wheres[3] = "intersect_"
            elif wheres[4][1] == 4 and wheres[2][1] == 5 and int(wheres[4][3]) < int(wheres[2][3]):
                wheres[3] = "intersect_"
            elif wheres[4][1] == 5 and wheres[2][1] == 4 and int(wheres[4][3]) > int(wheres[2][3]):
                wheres[3] = "intersect_"
            elif wheres[4][1] == 3 and wheres[2][1] == 6 and int(wheres[4][3]) > int(wheres[2][3]):
                wheres[3] = "intersect_"
            elif wheres[4][1] == 6 and wheres[2][1] == 3 and int(wheres[4][3]) < int(wheres[2][3]):
                wheres[3] = "intersect_"
            return wheres
    except:
        pass
    return wheres


def infer_group_for_exact_match(group_list, table_json):
    """V1.2
    infer the intersect
    """
    table = set()
    for g in group_list:
        table.add(g.split(".")[0])
    if len(table) == 1:
        table = list(table)[0]
        t_idx = [n.lower() for n in table_json['table_names_original']].index(table.lower())
        pk = primary_keys(table_json, t_idx)
        if pk >= 0:
            return [table_json['table_column_names_original'][pk][1]]
    return group_list


def inference_sql(sql_dict, table_json, args, join_on_label=None):
    """
    modified from NatSQL V1.0 inference code.
    """
    global globe_join_on_label_count
    globe_join_on_label_count = 0
    sql_dict['where'] = intersect_check(sql_dict['where'])
    groupby_list = []
    groupby_top = ""
    re_sql = "select distinct " if sql_dict['select'][0] else "select "
    orderby_sql, table_list, agg_in_order = ("", [], False)
    # if args.orderby_to_subquery and is_orderby_for_subquery(sql_dict): # v1.1
    #     orderby_sql,table_list,agg_in_order = ("",[],False)
    # else:
    #     orderby_sql,table_list,agg_in_order = create_order_by(sql_dict['orderBy'],sql_dict['limit'])

    # Get table info from select column
    for column in sql_dict['select'][1]:
        table = column[1][1][1].split('.')[0].lower()
        if not table in table_list:
            table_list.append(table)
        select_unit = select_unit_back(column)
        if not (column[0] or column[1][1][0]):
            groupby_list.append(select_unit)
        re_sql += select_unit + ' , '
    re_sql = re_sql[:-3]
    top_select_table_list = copy.deepcopy(table_list)
    # Add table info to select column
    break_idx, table_list, next_sql, sql_where, sql_having, orderby_sql_, next_table_list = get_where_column(sql_dict,
                                                                                                             table_list,
                                                                                                             0, SQL_TOP,
                                                                                                             table_json,
                                                                                                             args)
    if break_idx < 0 or next_sql == SQL_TOP:
        orderby_sql, table_list_order, agg_in_order = create_order_by(sql_dict['orderBy'], sql_dict['limit'])
        for order_t in table_list_order:
            if order_t.lower() not in table_list:
                table_list.append(order_t.lower())

    if sql_dict['groupBy']:  # V1.1:
        groupby_top = " group by " + col_unit_back(sql_dict['groupBy'][0])
    elif (len(groupby_list) != len(sql_dict['select'][1]) and groupby_list) or sql_having.strip() != '' or (
            agg_in_order and groupby_list) or orderby_sql_.strip():
        if args.group_for_exact_match and len(groupby_list) > 1:
            groupby_list = infer_group_for_exact_match(groupby_list, table_json)
        groupby_top = " group by " + ",".join(groupby_list)

    orderby_sql += orderby_sql_
    from_table_net, table_fk_list = get_table_network(table_json, table_list, join_on_label)
    top_sql_list = [re_sql]
    re_sql += create_from_table(from_table_net, table_json['table_names_original'],
                                table_json['table_column_names_original'], table_fk_list)
    top_sql_list.append(re_sql + sql_where + groupby_top + sql_having)

    if sql_dict['where']:
        while next_sql:
            table_list = next_table_list  # []#V1.2
            if next_sql == SQL_TOP:
                sub_sql = " " + sql_dict['where'][break_idx][:-1] + " " + top_sql_list[0]
                table_list = top_select_table_list
                start_new_top_sql = True
            else:
                select_column = col_unit_back(sql_dict['where'][break_idx][3])
                sub_sql = "select " + select_column
                if sql_dict['where'][break_idx][3][1].split('.')[0].lower() not in table_list:
                    table_list.append(sql_dict['where'][break_idx][3][1].split('.')[0].lower())
                start_new_top_sql = False

            break_idx, table_list, next_sql, sql_where, sql_having, orderby_sql_, next_table_list = get_where_column(
                sql_dict, table_list, break_idx + 1, next_sql, table_json, args)
            if args.orderby_to_subquery and not orderby_sql_:
                orderby_sql_, table_list = orderby_to_subquery(sql_dict, table_list)  # v1.1

            # if not start_new_top_sql:
            from_table_net, table_fk_list = get_table_network(table_json, table_list, join_on_label)
            sub_sql += create_from_table(from_table_net, table_json['table_names_original'],
                                         table_json['table_column_names_original'], table_fk_list)

            # if sql_where.strip() != 'where':
            sub_sql += sql_where

            if not start_new_top_sql:
                # if (sql_having.strip() and select_column) or (orderby_sql_.strip() and select_column):#v1.0
                if (sql_having.strip() and select_column) or ((
                                                                      "max(" in orderby_sql_ or "min(" in orderby_sql_ or "count(" in orderby_sql_ or "sum(" in orderby_sql_ or "avg(" in orderby_sql_) and select_column):  # v1.0
                    sub_sql += " group by " + select_column
            else:
                if groupby_top.strip():
                    sub_sql += groupby_top
                elif (sql_having.strip() != '' and groupby_list) or (orderby_sql_.strip() and groupby_list):
                    sub_sql += " group by " + ",".join(groupby_list)

            sub_sql += sql_having + orderby_sql_

            if start_new_top_sql:
                top_sql_list.append(sub_sql)
            else:
                top_sql_list[len(top_sql_list) - 1] = top_sql_list[len(top_sql_list) - 1].replace('@@@', sub_sql, 1)

    re_sql = ""
    for idx, sql in enumerate(top_sql_list):
        if idx > 0:
            re_sql += sql

    re_sql += orderby_sql

    return re_sql


def from_net_to_str(table_json, from_table_netss):
    str_all = []
    for fts in from_table_netss:
        str_list = []
        for ft in fts:
            str_list.append(
                create_from_table(ft, table_json['table_names_original'], table_json['table_column_names_original'],
                                  []))
        str_all.append(str_list)
    return str_all


def search_all_join_on(sql_dict, table_json, args, join_on_label=None):
    """
    modified from NatSQL V1.0 inference code.
    """
    all_from = []
    global globe_join_on_label_count
    globe_join_on_label_count = 0

    sql_dict['where'] = intersect_check(sql_dict['where'])
    groupby_list = []
    groupby_top = ""
    re_sql = "select distinct " if sql_dict['select'][0] else "select "
    orderby_sql, table_list, agg_in_order = ("", [], False)
    # if args.orderby_to_subquery and is_orderby_for_subquery(sql_dict): # v1.1
    #     orderby_sql,table_list,agg_in_order = ("",[],False)
    # else:
    #     orderby_sql,table_list,agg_in_order = create_order_by(sql_dict['orderBy'],sql_dict['limit'])

    # Get table info from select column
    for column in sql_dict['select'][1]:
        table = column[1][1][1].split('.')[0].lower()
        if not table in table_list:
            table_list.append(table)
        select_unit = select_unit_back(column)
        if not (column[0] or column[1][1][0]):
            groupby_list.append(select_unit)
        re_sql += select_unit + ' , '
    re_sql = re_sql[:-3]
    top_select_table_list = copy.deepcopy(table_list)
    # Add table info to select column
    break_idx, table_list, next_sql, sql_where, sql_having, orderby_sql_, next_table_list = get_where_column(sql_dict,
                                                                                                             table_list,
                                                                                                             0, SQL_TOP,
                                                                                                             table_json,
                                                                                                             args)
    if break_idx < 0 or next_sql == SQL_TOP:
        orderby_sql, table_list_order, agg_in_order = create_order_by(sql_dict['orderBy'], sql_dict['limit'])
        for order_t in table_list_order:
            if order_t.lower() not in table_list:
                table_list.append(order_t.lower())

    if sql_dict['groupBy']:  # V1.1:
        groupby_top = " group by " + col_unit_back(sql_dict['groupBy'][0])
    elif (len(groupby_list) != len(sql_dict['select'][1]) and groupby_list) or sql_having.strip() != '' or (
            agg_in_order and groupby_list) or orderby_sql_.strip():
        if args.group_for_exact_match and len(groupby_list) > 1:
            groupby_list = infer_group_for_exact_match(groupby_list, table_json)
        groupby_top = " group by " + ",".join(groupby_list)

    orderby_sql += orderby_sql_
    from_table_net, table_fk_list = get_table_network(table_json, table_list, join_on_label)

    from_table_netss, _ = get_table_network(table_json, table_list, join_on_label, False)
    all_from.append(from_table_netss)

    top_sql_list = [re_sql]
    re_sql += create_from_table(from_table_net, table_json['table_names_original'],
                                table_json['table_column_names_original'], table_fk_list)
    top_sql_list.append(re_sql + sql_where + groupby_top + sql_having)

    if sql_dict['where']:
        while next_sql:
            table_list = next_table_list  # []#V1.2
            if next_sql == SQL_TOP:
                sub_sql = " " + sql_dict['where'][break_idx][:-1] + " " + top_sql_list[0]
                table_list = top_select_table_list
                start_new_top_sql = True
            else:
                select_column = col_unit_back(sql_dict['where'][break_idx][3])
                sub_sql = "select " + select_column
                if sql_dict['where'][break_idx][3][1].split('.')[0].lower() not in table_list:
                    table_list.append(sql_dict['where'][break_idx][3][1].split('.')[0].lower())
                start_new_top_sql = False

            break_idx, table_list, next_sql, sql_where, sql_having, orderby_sql_, next_table_list = get_where_column(
                sql_dict, table_list, break_idx + 1, next_sql, table_json, args)
            if args.orderby_to_subquery and not orderby_sql_:
                orderby_sql_, table_list = orderby_to_subquery(sql_dict, table_list)  # v1.1

            # if not start_new_top_sql:
            from_table_net, table_fk_list = get_table_network(table_json, table_list, join_on_label)
            from_table_netss, _ = get_table_network(table_json, table_list, join_on_label, False)
            all_from.append(from_table_netss)
            sub_sql += create_from_table(from_table_net, table_json['table_names_original'],
                                         table_json['table_column_names_original'], table_fk_list)

            # if sql_where.strip() != 'where':
            sub_sql += sql_where

            if not start_new_top_sql:
                # if (sql_having.strip() and select_column) or (orderby_sql_.strip() and select_column):#v1.0
                if (sql_having.strip() and select_column) or ((
                                                                      "max(" in orderby_sql_ or "min(" in orderby_sql_ or "count(" in orderby_sql_ or "sum(" in orderby_sql_ or "avg(" in orderby_sql_) and select_column):  # v1.0
                    sub_sql += " group by " + select_column
            else:
                if groupby_top.strip():
                    sub_sql += groupby_top
                elif (sql_having.strip() != '' and groupby_list) or (orderby_sql_.strip() and groupby_list):
                    sub_sql += " group by " + ",".join(groupby_list)

            sub_sql += sql_having + orderby_sql_

            if start_new_top_sql:
                top_sql_list.append(sub_sql)
            else:
                top_sql_list[len(top_sql_list) - 1] = top_sql_list[len(top_sql_list) - 1].replace('@@@', sub_sql, 1)

    re_sql = ""
    for idx, sql in enumerate(top_sql_list):
        if idx > 0:
            re_sql += sql

    re_sql += orderby_sql

    return re_sql, all_from, sql_dict

# def downgrade(sql_dict, table_json, args, join_on_label=None):
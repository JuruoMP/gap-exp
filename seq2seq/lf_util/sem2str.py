from seq2seq.third_party.test_suite.process_sql import get_sql



def sem2str(d, schema):
    return ''

def str2sem(s, schems):
    def tables_to_view(table_schema):
        return table_schema

    db_table_view = tables_to_view(db_table)
    sem_dict = get_sql(db_table_view, sql)
    return {}
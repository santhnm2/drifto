from datetime import datetime, timedelta
from random import randrange, random
from pyarrow import json
from pyarrow import parquet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", type=int, help="number of users", default=100)
parser.add_argument("-w", type=int, help="number of weeks", default=100)
parser.add_argument("-a", type=int, help="actions per week", default=10)
args = parser.parse_args()
num_users = args.u
num_weeks = args.w
actions_per_week = args.a

actions = ['purchase', 'email_opened', 'page_visited', 'search_performed']

start_date = datetime.strptime("12.14.2020", "%m.%d.%Y")
pages = ['/bug', '/crypto', '/ai', '/payments', '/home']
bug_odds = 0.1

saw_bug_last_week = set()
saw_bug_this_week = set()

e = open('events.parquet', 'w')
t = open('transactions.parquet', 'w')

for i in range(num_weeks):
    for j in range(num_users):
        bug_possible = random() < bug_odds
        for k in range(actions_per_week):
            first_possible_action = 1 if j in saw_bug_last_week else 0
            action = actions[randrange(first_possible_action, len(actions))]
            timestamp = str(start_date + timedelta(weeks=i) + \
                timedelta(seconds=randrange(60 * 60 * 24 * 7)))
            row = '{"user_id": %d, "timestamp": "%s", "action": "%s"' % \
                (j, timestamp, action)
            if action == 'purchase':
                row = '{"user_id": %d, "timestamp": "%s", "order_total": %d}' % \
                    (j, timestamp, randrange(1, 1000))
                t.write(row + '\n')
                continue
            elif action == 'page_visited':
                first_possible_page = 0 if bug_possible else 1
                page = pages[randrange(first_possible_page, len(pages))]
                if page == '/bug':
                    saw_bug_this_week.add(j)
                row += ', "attributes": "{\\\"page\\\": \\\"%s\\\"}"}' % page
            else:
                row += ', "attributes": "{}"}'
            e.write(row + '\n')
    saw_bug_last_week = saw_bug_this_week
    saw_bug_this_week = set()

e.close()
t.close()
for f in ['events.parquet', 'transactions.parquet']:
    parquet.write_table(json.read_json(f), f)

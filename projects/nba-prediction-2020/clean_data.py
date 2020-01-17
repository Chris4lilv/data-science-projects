import csv
from collections import defaultdict


RAW_FILE = 'Result.csv'

def csvDictWriter(input):
    with open('tempData.csv', 'w') as f:
        fieldnames = ['Visitor', 'Home']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in input:
            writer.writerow({'Visitor':key})

def writer(input):
    with open('tempData.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(input)

def giveGameResult(input):
    list = []
    with open(input) as r:
        reader = csv.DictReader(r)
        for row in reader:
            home = int(row['PTS1'])
            visit = int(row['PTS2'])
            if home < visit:
                list.append('L')
            else:
                list.append('W')
    return list

def winnerHomeOrVisitor(input):
    list = []
    with open(input) as r:
        reader = csv.DictReader(r)
        for row in reader:
            if row['Result'] == 'L':
                list.append('H')
            else:
                list.append('V')
    return list

def win_loss(input):
    win_loss = {}
    with open(input, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            homeTeam = row['Home']
            visitTeam = row['Visitor']
            if row['WLoc'] == 'H':
                d[homeTeam] = visitTeam
                print(d)
            else:
                d[visitTeam] = homeTeam
    return win_loss

def readCSV(input):
    with open(input) as csvfile:
        readCSV = csv.reader(csvfile)
        Homes = []
        Visitors = []
        for row in readCSV:
            visitor = row[2]
            home = row[4]

            Homes.append(home)
            Visitors.append(visitor)
        print(Homes)
    return Homes, Visitors
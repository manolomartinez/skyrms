"""
The script that calculates basins for all games
"""
import os
import calculations as c

def main():
    basedir = os.getcwd()
    gamefiles = os.listdir('gamesperc/')
    for gamefile in gamefiles:
        print(gamefile, '...')
        filepath = os.path.join(basedir, 'gamesperc', gamefile)
        dirpath = os.path.join(basedir, 'results', gamefile)
        os.mkdir(dirpath)
        c.one_batch(filepath, dirpath)
        print('Done')

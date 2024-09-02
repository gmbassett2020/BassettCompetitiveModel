# Develop simulated football season.
# Connect to a Java Neural Network using Lasagne.
# http://lasagne.readthedocs.org/
#    More in-depth examples and reproductions of paper results are maintained in
#    a separate repository: https://github.com/Lasagne/Recipes
# 2022-01-17
# 2022-08-05
#  Moved from jupyter-notebook to python file.  
#  Removing neural net items since focusing first on purely linear model.

import math
from scipy import special

import random
   # Syntax : random.gauss(mu, sigma)
   #
   # Parameters :
   # mu : mean
   # sigma : standard deviation
   #
   # Returns : a random gaussian distribution floating number
   #
   # If we're trying to establish equivalency between RMS and standard deviation, .... if the mean is zero, as is often 
   # the case in electrical signals, there is no difference between the RMS calculation and the 
   # standard-deviation calculation.Jul 28, 2020

import sys

import pandas

#
# Settings
#

# Default values

#useTeamListFile = "colI-list-1998_2019.csv"
#inputPowerFiles = "colI-fitPowers-1998_2019-s20230805-f0-noForce.csv,colI-fitPowers-1998_2019-s20230805-f1-noForce.csv,colI-fitPowers-1998_2019-s20230805-f2-noForce.csv,colI-fitPowers-1998_2019-s20230805-f3-noForce.csv,colI-fitPowers-1998_2019-s20230805-f4-noForce.csv"
#outputPowerFile = "colI-averagePowers-1998_2019-s20230805-noForce.csv"
useTeamListFile = "nfl-list-2001_2022.csv"
inputPowerFiles = "nfl-fitPowers-2001_2019-s20230904-f0-noForce.csv,nfl-fitPowers-2001_2019-s20230904-f1-noForce.csv,nfl-fitPowers-2001_2019-s20230904-f2-noForce.csv,nfl-fitPowers-2001_2019-s20230904-f3-noForce.csv,nfl-fitPowers-2001_2019-s20230904-f4-noForce.csv"
outputPowerFile = "nfl-averagePowers-2001_2019-s20230904-noForce.csv"

verbose = False

print("ARGV: "+str(sys.argv))

# Overide with command line arguents
if (len(sys.argv) > 1):
   for iArg in range(1,len(sys.argv)):
      keyAndValueString = sys.argv[iArg].split("=",2)

#     if (keyAndValueString[0] == "seasonRandomSeed"):
#        seasonRandomSeed = int(keyAndValueString[1])
#        print("Setting seasonRandomSeed to "+str(seasonRandomSeed))

#     elif (keyAndValueString[0] == "deprioritizeTotalForCost"):
#        deprioritizeTotalForCost = float(keyAndValueString[1])
#        print("Setting deprioritizeTotalForCost to "+str(deprioritizeTotalForCost))

#     elif (keyAndValueString[0] == "optimizeWeekWeights"):
#        optimizeWeekWeights = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
#        print("Setting optimizeWeekWeights to "+str(optimizeWeekWeights))

      if (keyAndValueString[0] == "inputPowerFiles"):
         inputPowerFiles = keyAndValueString[1]
         print("Loading fit powers from "+inputPowerFiles)

      elif (keyAndValueString[0] == "outputPowerFile"):
         outputPowerFile = keyAndValueString[1]
         print("Saving average power fit to "+outputPowerFile)

      elif (keyAndValueString[0] == "useTeamListFile"):
         useTeamListFile = keyAndValueString[1]
         print("Loading team info from "+useTeamListFile)

      else:
         raise Exception("ERROR: unknown parameter "+sys.argv[iArg])

inputPowerFilesArray = inputPowerFiles.split(',')

print("Input Power Files: "+str(inputPowerFilesArray))

teamList = pandas.read_csv(useTeamListFile)

divisionByTeamAndYear = {}

for listIndex in teamList.index:
   teamName = teamList["Team"][listIndex]
   division = teamList["Division"][listIndex]
   year = teamList["Year"][listIndex]
   if (divisionByTeamAndYear.get(teamList["Team"][listIndex]) == None):
      divisionByTeamAndYear[teamName] = {}
   divisionByTeamAndYear[teamName][year] = division

powersByTeamAndSeason = {}
powersByDivision = {}
for inputCsv in inputPowerFilesArray:
   powers = pandas.read_csv(inputCsv)
   for powerIndex in powers.index:
      teamName = powers["teamName"][powerIndex]
      teamId = powers["teamId"][powerIndex]
      year = powers["year"][powerIndex]
      offense = powers["offense"][powerIndex]
      defense = powers["defense"][powerIndex]
      division = divisionByTeamAndYear[teamName].get(year)
      if (division != None): # skip years where team not in a division
         # division powers
         if (powersByDivision.get(division) == None):
            powersByDivision[division] = {}
            powersByDivision[division]["count"] = 0
            powersByDivision[division]["offenseTotal"] = 0
            powersByDivision[division]["defenseTotal"] = 0
            powersByDivision[division]["countRms"] = 0
            powersByDivision[division]["offenseRms"] = 0
            powersByDivision[division]["defenseRms"] = 0
         # team powers
         if (powersByTeamAndSeason.get(teamId) == None):
            powersByTeamAndSeason[teamId] = {}
         if (powersByTeamAndSeason[teamId].get(year) == None):
            powersByTeamAndSeason[teamId][year] = {}
            powersByTeamAndSeason[teamId][year]["teamName"] = teamName
            powersByTeamAndSeason[teamId][year]["offenseTotal"] = 0
            powersByTeamAndSeason[teamId][year]["defenseTotal"] = 0
            powersByTeamAndSeason[teamId][year]["count"] = 0
            powersByTeamAndSeason[teamId][year]["division"] = division
            powersByTeamAndSeason[teamId][year]["offenseArray"] = []
            powersByTeamAndSeason[teamId][year]["defenseArray"] = []
         powersByTeamAndSeason[teamId][year]["offenseArray"].append(offense)
         powersByTeamAndSeason[teamId][year]["defenseArray"].append(defense)
         powersByTeamAndSeason[teamId][year]["offenseTotal"] += offense
         powersByTeamAndSeason[teamId][year]["defenseTotal"] += defense
         powersByTeamAndSeason[teamId][year]["count"] += 1

# division average, team average and RMS
for teamId in powersByTeamAndSeason.keys():
   # RMS over fits
   for year in sorted(powersByTeamAndSeason[teamId].keys()):
      powersByTeamAndSeason[teamId][year]["offenseAverage"] = powersByTeamAndSeason[teamId][year]["offenseTotal"]/powersByTeamAndSeason[teamId][year]["count"]
      powersByTeamAndSeason[teamId][year]["defenseAverage"] = powersByTeamAndSeason[teamId][year]["defenseTotal"]/powersByTeamAndSeason[teamId][year]["count"]
      division = powersByTeamAndSeason[teamId][year]["division"]
      powersByDivision[division]["count"] += 1
      powersByDivision[division]["offenseTotal"] += powersByTeamAndSeason[teamId][year]["offenseAverage"]
      powersByDivision[division]["defenseTotal"] += powersByTeamAndSeason[teamId][year]["defenseAverage"]
      powersByTeamAndSeason[teamId][year]["offenseRms"] = 0
      powersByTeamAndSeason[teamId][year]["defenseRms"] = 0
      for offense in powersByTeamAndSeason[teamId][year]["offenseArray"]:
         powersByDivision[division]["countRms"] += 1
         offenseDiffSq = (offense - powersByTeamAndSeason[teamId][year]["offenseAverage"])**2
         powersByTeamAndSeason[teamId][year]["offenseRms"] += offenseDiffSq
      for defense in powersByTeamAndSeason[teamId][year]["defenseArray"]:
         defenseDiffSq = (defense - powersByTeamAndSeason[teamId][year]["defenseAverage"])**2
         powersByTeamAndSeason[teamId][year]["defenseRms"] += defenseDiffSq
      powersByTeamAndSeason[teamId][year]["offenseRms"] = math.sqrt(powersByTeamAndSeason[teamId][year]["offenseRms"]/powersByTeamAndSeason[teamId][year]["count"])
      powersByTeamAndSeason[teamId][year]["defenseRms"] = math.sqrt(powersByTeamAndSeason[teamId][year]["defenseRms"]/powersByTeamAndSeason[teamId][year]["count"])

# division average
for division in sorted(powersByDivision.keys()):
   powersByDivision[division]["offenseRms"] = 0
   powersByDivision[division]["defenseRms"] = 0
   powersByDivision[division]["countRms"] = 0
   powersByDivision[division]["offenseChangeRms"] = 0
   powersByDivision[division]["defenseChangeRms"] = 0
   powersByDivision[division]["countChange"] = 0
   powersByDivision[division]["offenseAverage"] = powersByDivision[division]["offenseTotal"] / powersByDivision[division]["count"]
   powersByDivision[division]["defenseAverage"] = powersByDivision[division]["defenseTotal"] / powersByDivision[division]["count"]
   
# division RMS
for teamId in powersByTeamAndSeason.keys():
   for year in sorted(powersByTeamAndSeason[teamId].keys()):
      division = powersByTeamAndSeason[teamId][year]["division"]
      powersByDivision[division]["countRms"] += 1
      powersByDivision[division]["offenseRms"] += (powersByDivision[division]["offenseAverage"] - powersByTeamAndSeason[teamId][year]["offenseAverage"])**2
      powersByDivision[division]["defenseRms"] += (powersByDivision[division]["defenseAverage"] - powersByTeamAndSeason[teamId][year]["defenseAverage"])**2

# division average
for division in sorted(powersByDivision.keys()):
   powersByDivision[division]["offenseRms"] = math.sqrt(powersByDivision[division]["offenseRms"]/powersByDivision[division]["countRms"])
   powersByDivision[division]["defenseRms"] = math.sqrt(powersByDivision[division]["defenseRms"]/powersByDivision[division]["countRms"])

# division RMS and season change RMS
for teamId in powersByTeamAndSeason.keys():
   for year in sorted(powersByTeamAndSeason[teamId].keys()):
      division = powersByTeamAndSeason[teamId][year]["division"]
      powersByDivision[division]["offenseRms"] += (powersByDivision[division]["offenseAverage"] - powersByTeamAndSeason[teamId][year]["offenseAverage"])**2
      powersByDivision[division]["defenseRms"] += (powersByDivision[division]["defenseAverage"] - powersByTeamAndSeason[teamId][year]["defenseAverage"])**2
      powersByDivision[division]["countRms"] += 1
      if (teamId == 24):
         print("::: teamID="+str(teamId)+" year="+str(year)+" div="+division)

   firstSeason = True
   for year in sorted(powersByTeamAndSeason[teamId].keys()):
      if (firstSeason):
         firstSeason = False
         if (teamId == 24):
            print("::: year="+str(year)+" first")
      else:
         powersByDivision[division]["offenseChangeRms"] += (offensePrevious - powersByTeamAndSeason[teamId][year]["offenseAverage"])**2
         powersByDivision[division]["defenseChangeRms"] += (defensePrevious - powersByTeamAndSeason[teamId][year]["defenseAverage"])**2
         if (teamId == 24):
            print("::: year="+str(year)+" offChangeWIP="+str(powersByDivision[division]["offenseChangeRms"])+" prev="+str(offensePrevious)+" aveForYear="+str(powersByTeamAndSeason[teamId][year]["offenseAverage"]))
         powersByDivision[division]["countChange"] += 1
      offensePrevious = powersByTeamAndSeason[teamId][year]["offenseAverage"]
      defensePrevious = powersByTeamAndSeason[teamId][year]["defenseAverage"]

for division in powersByDivision.keys():
   powersByDivision[division]["offenseRms"] = math.sqrt(powersByDivision[division]["offenseRms"] / powersByDivision[division]["countRms"])
   powersByDivision[division]["defenseRms"] = math.sqrt(powersByDivision[division]["defenseRms"] / powersByDivision[division]["countRms"])
   powersByDivision[division]["offenseChangeRms"] = math.sqrt(powersByDivision[division]["offenseChangeRms"] / powersByDivision[division]["countChange"])
   powersByDivision[division]["defenseChangeRms"] = math.sqrt(powersByDivision[division]["defenseChangeRms"] / powersByDivision[division]["countChange"])
   print("Division "+division
      +" offAve "+str(powersByDivision[division]["offenseAverage"])
      +" defAve "+str(powersByDivision[division]["defenseAverage"])
      +" offRms "+str(powersByDivision[division]["offenseRms"])
      +" defRms "+str(powersByDivision[division]["defenseRms"])
      +" offDelta "+str(powersByDivision[division]["offenseChangeRms"])
      +" defDelta "+str(powersByDivision[division]["defenseChangeRms"]))
         

teamNames = []
teamIds = []
seasons = []
offense = []
defense = []
offenseRms = []
defenseRms = []
   
for teamId in sorted(powersByTeamAndSeason.keys()):
   for year in sorted(powersByTeamAndSeason[teamId].keys()):
      teamNames.append(powersByTeamAndSeason[teamId][year]["teamName"])
      teamIds.append(teamId)
      seasons.append(year)
      offense.append(powersByTeamAndSeason[teamId][year]["offenseAverage"])
      defense.append(powersByTeamAndSeason[teamId][year]["defenseAverage"])
      offenseRms.append(powersByTeamAndSeason[teamId][year]["offenseRms"])
      defenseRms.append(powersByTeamAndSeason[teamId][year]["defenseRms"])

teamPowers = {'teamName': teamNames, 'teamId': teamIds, 'year': seasons, 'offense': offense, 'defense': defense, 'offenseRms':offenseRms, 'defenseRms':defenseRms}
teamPowersPandas = pandas.DataFrame(teamPowers)
teamPowersPandas.to_csv(outputPowerFile)


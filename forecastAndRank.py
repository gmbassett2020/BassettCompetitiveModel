# Develop simulated football season.
# Connect to a Java Neural Network using Lasagne.
# http://lasagne.readthedocs.org/
#    More in-depth examples and reproductions of paper results are maintained in
#    a separate repository: https://github.com/Lasagne/Recipes
# 2022-01-17
# 2022-08-05
#  Moved from jupyter-notebook to python file.  
#  Removing neural net items since focusing first on purely linear model.
# 2023-08-08
#  Run one season, either simulated or from actual data.

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

# Season classes

import team
import game
import gameSeason
import gameSimulator

#
# Settings
#

# Default values

currentSeason = 2023 # This is the season on which to do predictions, results and rankings.
# col
#maxRound = 3 # This is the round (1 based) on which to do predictions.  Results are done for round before this.
# nfl
maxRound = 1 # This is the round (1 based) on which to do predictions.  Results are done for round before this.

seasonRandomSeed = 20210905
fitRandomSeed = 20210905
learningRate = 0.01
numberOfIterationsForEachWeek = 150
#numberOfIterationsForEachWeek = 5

verbose = False

adamUpdateEnable = True

spreadRms = 14
rmsMultiplier = 1.0
roundScores = False

#currentSeason = 2000

#useTeamListFile = "colI-list-1998_2019.csv"
#useTeamSeasonsFile = "colI-seasons-1998_2019.csv"
# col
#useTeamListFile = "colI-list-2020_2023.csv"
#useTeamSeasonsFile = "colI-seasons-2020_2023.csv"
# nfl
useTeamListFile = "nfl-list-2001_2023.csv"
useTeamSeasonsFile = "nfl-seasons-2001_2023.csv"

#useTeamListFile = None
#useTeamSeasonsFile = None
teamList = None
teamSeasons = None
#saveTeamPowersToFile = None
# col
#saveTeamPowersToFile = "colI-powers-2023.csv"
# nfl
saveTeamPowersToFile = "nfl-powers-2023.csv"

outputFile = None

fitTeamPowers = True

usePowerUncertainty = False

deprioritizeTotalForCost = 0.25

#2023-08-08
# found linear parameters with this (averaged over the 5 cases):
# spdCoefOffInit=8.08 spdCoefDefInit=5.93 totCoefOffInit=2.76 totCoefDefInit=-3.28 totCoef0Init=28.89 

spdCoefOffInit = 8.08
spdCoefDefInit = 5.93
totCoefOffInit = 2.76
totCoefDefInit = -3.28
totCoef0Init = 28.89 

adamUpdateEnable = True # use Adam - a method for stochasitc optimization.  https://arxiv.org/pdf/1412.6980.pdf

print("ARGV: "+str(sys.argv))

# Overide with command line arguents
if (len(sys.argv) > 1):
   for iArg in range(1,len(sys.argv)):
      keyAndValueString = sys.argv[iArg].split("=",2)

      if (keyAndValueString[0] == "seasonRandomSeed"):
         seasonRandomSeed = int(keyAndValueString[1])
         print("Setting seasonRandomSeed to "+str(seasonRandomSeed))

      elif (keyAndValueString[0] == "fitRandomSeed"):
         fitRandomSeed = int(keyAndValueString[1])
         print("Setting fitRandomSeed to "+str(fitRandomSeed))

      elif (keyAndValueString[0] == "learningRate"):
         learningRate = float(keyAndValueString[1])
         print("Setting learningRate to "+str(learningRate))

#     elif (keyAndValueString[0] == "numberOfSeasons"):
#        numberOfSeasons = int(keyAndValueString[1])
#        print("Setting numberOfSeasons to "+str(numberOfSeasons))

      elif (keyAndValueString[0] == "maxRound"):
         maxRound = int(keyAndValueString[1])
         print("Setting maxRound to "+str(maxRound))

      elif (keyAndValueString[0] == "numberOfIterationsForWeekWeights"):
         numberOfIterationsForWeekWeights = int(keyAndValueString[1])
         print("Setting numberOfIterationsForWeekWeights to "+str(numberOfIterationsForWeekWeights))

      elif (keyAndValueString[0] == "deprioritizeTotalForCost"):
         deprioritizeTotalForCost = float(keyAndValueString[1])
         print("Setting deprioritizeTotalForCost to "+str(deprioritizeTotalForCost))

      elif (keyAndValueString[0] == "currentSeason"):
         currentSeason = int(keyAndValueString[1])
         print("Setting currentSeason to "+str(currentSeason))

      elif (keyAndValueString[0] == "verbose"):
         verbose = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting verbose to "+str(verbose))

      elif (keyAndValueString[0] == "adamUpdateEnable"):
         adamUpdateEnable = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting adamUpdateEnable to "+str(adamUpdateEnable))

      elif (keyAndValueString[0] == "useTeamListFile"):
         useTeamListFile = keyAndValueString[1]
         print("Loading team info from "+useTeamListFile)

      elif (keyAndValueString[0] == "useTeamSeasonsFile"):
         useTeamSeasonsFile = keyAndValueString[1]
         print("Loading season info from "+useTeamSeasonsFile)

      elif (keyAndValueString[0] == "saveTeamPowersToFile"):
         saveTeamPowersToFile = keyAndValueString[1]
         print("Saving team powers to "+saveTeamPowersToFile)

      elif (keyAndValueString[0] == "outputFile"):
         outputFile = keyAndValueString[1]
         print("Saving results, predictions and statistics to "+outputFile)

      elif (keyAndValueString[0] == "spdCoefOffInit"):
         spdCoefOffInit = float(keyAndValueString[1])
         print("Setting spdCoefOffInit to "+str(spdCoefOffInit))

      elif (keyAndValueString[0] == "spdCoefDefInit"):
         spdCoefDefInit = float(keyAndValueString[1])
         print("Setting spdCoefDefInit to "+str(spdCoefDefInit))

      elif (keyAndValueString[0] == "totCoefOffInit"):
         totCoefOffInit = float(keyAndValueString[1])
         print("Setting totCoefOffInit to "+str(totCoefOffInit))

      elif (keyAndValueString[0] == "totCoefDefInit"):
         totCoefDefInit = float(keyAndValueString[1])
         print("Setting totCoefDefInit to "+str(totCoefDefInit))

      elif (keyAndValueString[0] == "totCoef0Init"):
         totCoef0Init = float(keyAndValueString[1])
         print("Setting totCoef0Init to "+str(totCoef0Init))

      elif (keyAndValueString[0] == "usePowerUncertainty"):
         usePowerUncertainty = (keyAndValueString[1] == "True" or keyAndValueString[1] == "true")
         print("Setting usePowerUncertainty to "+str(usePowerUncertainty))

      else:
         print("ERROR: unknown parameter "+keyAndValueString[0])
         raise Exception("ERROR: unknown parameter")

random.seed(seasonRandomSeed)

#
# Set initial values for linear parameters
#

homeAdvantage = 3.
#first set
#spdCoefOffInit = 18.4
#spdCoefDefInit = 17.6 
#totCoef0Init = -17 
#totCoefOffInit = 14.8 
#totCoefDefInit = -9.6 
#rmsCoef0Init = rmsMultiplier * 16.173 
#rmsCoefOffInit = rmsMultiplier * 1.324 
#rmsCoefDefInit = rmsMultiplier * -1.684
# wip
#spdCoefOffInit = 18.4
#spdCoefDefInit = 25.8
#totCoef0Init = 0
#totCoefOffInit = 14.8 
#totCoefDefInit = -9.6 
#rmsCoef0Init = rmsMultiplier * 16.173 
#rmsCoefOffInit = rmsMultiplier * 1.324 
#rmsCoefDefInit = rmsMultiplier * -1.684
maxConferenceGameWeights = 10

gameSimulator = gameSimulator.GameSimulator(homeAdvantage, spdCoefOffInit, spdCoefDefInit, totCoef0Init, totCoefOffInit, totCoefDefInit, roundScores, spreadRms*rmsMultiplier, maxConferenceGameWeights)

#2023-08-21
#Found optimized weights using the following:
#numberOfSeasons = 100
#numberOfIterationsForEachWeek = 300
#numberOfIterationsForWeekWeights = 300
# iWeekWeightIter 299 cost [2.540729170488716, 3.2494912466686587, 2.9159332983007564, 2.492151096500357, 2.078530999979104, 1.6805707654898485, 1.3180204509279119, 0.9948134722125765, 0.7095372492187236, 0.4895885685340341, 0.31366504326693634, 0.23683016621781747, 0.18779311163478776, 0, 0, 0, 0, 0, 0, 0]
# iWeekWeightIter 299 weight [0.8140645979562876, 0.6598250349845416, 0.6760580664368508, 0.7159214887274042, 0.7564336711722774, 0.7984806907197232, 0.8401389606679754, 0.8776309277658789, 0.910508970314625, 0.9389177480378471, 0.9601690539204841, 0.9689983845809068, 0.9903137951598799, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#2023-08-22
#New weights after removing rounding from predictions and adjusting initialization
# iWeekWeightIter 299 cost [6.061021605744927, 4.721845012974696, 3.643352469778301, 2.8629141073363282, 2.25540330871929, 1.74958077168523, 1.3309976628132625, 0.9841611102952907, 0.6954937947594652, 0.47059225292020385, 0.2938697212488857, 0.21901620500612098, 0.17622912277768846, 0, 0, 0, 0, 0, 0, 0]
# iWeekWeightIter 299 weight [0.7941167790817569, 0.7195257962111826, 0.7574704435994234, 0.7948800055001095, 0.8262671256461386, 0.8557465042369405, 0.8813065655037028, 0.9058579695882067, 0.928092095456229, 0.9493606398415331, 0.9662703181070563, 0.9720919860088111, 0.9938275247806949, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
#3190.843u 0.326s 53:11.38 99.9% 0+0k 0+14048io 0pf+0w
gameSimulator.SetWeekWeightsToBestFit()

#
# Set up the seasons
#

gameSeasons = []
minimalFirstRound = False

if (useTeamListFile != None and useTeamSeasonsFile != None):
   teamList = pandas.read_csv(useTeamListFile)
   teamSeasons = pandas.read_csv(useTeamSeasonsFile)

   minimalFirstRound = True

   gameSeasonDict = {}
   roundsByYearDict = {}

   # Add all teams and seasons from the list file.
   for listIndex in teamList.index:
      if (roundsByYearDict.get(teamList["Year"][listIndex]) == None):
         roundsByYearDict[teamList["Year"][listIndex]] = {}
      if (gameSeasonDict.get(teamList["Year"][listIndex]) == None):
         print("Creating gameSeason for "+str(teamList["Year"][listIndex]))
         thisGameSeason = gameSeason.GameSeason(teamList["Year"][listIndex], gameSimulator, "gs")
         gameSeasonDict[teamList["Year"][listIndex]] = thisGameSeason
         print("Created gameSeason "+str(teamList["Year"][listIndex])+" with seasonYear "+str(thisGameSeason.seasonYear))
      teamObject = team.Team.teamObjectByName.get(teamList["Team"][listIndex])
      if (teamObject == None):
         teamObject = team.Team(teamList["Team"][listIndex])
      teamObject.SetConferenceAndDivision(teamList["Year"][listIndex], teamList["Conference"][listIndex], teamList["Division"][listIndex])

   for gameIndex in teamSeasons.index:
      thisGameSeason = gameSeasonDict.get(teamSeasons["Season"][gameIndex])
      team1Object = team.Team.teamObjectByName.get(teamSeasons["Team1"][gameIndex])
      team2Object = team.Team.teamObjectByName.get(teamSeasons["Team2"][gameIndex])
      year = teamSeasons["Season"][gameIndex]
      roundName = teamSeasons["RoundName"][gameIndex]
      roundNumber = teamSeasons["RoundNumber"][gameIndex]
      homeField = 0.5
      gameRowString = "Year="+str(year)+" team1="+teamSeasons["Team1"][gameIndex]+" team2="+teamSeasons["Team2"][gameIndex]+" roundName="+roundName+" roundNumber="+str(roundNumber)+" homeFlag="+teamSeasons["HomeFlag"][gameIndex]+" score1="+str(teamSeasons["Score1"][gameIndex])+" score2="+str(teamSeasons["Score2"][gameIndex])
      if (teamSeasons["HomeFlag"][gameIndex] == "team1Home"):
         homeField = 1.0
      elif (teamSeasons["HomeFlag"][gameIndex] == "team2Home"):
         homeField = 0.0
      if (thisGameSeason == None):
         print("ERROR: bad entry in seasons file (Season listed in seasons file which is not present in list file) or unknown year in gameRow "+gameRowString)
         exit(1)
      if (team1Object == None):
         print("WARNING: skipping game because of unknown Team1="+str(teamSeasons["Team1"][gameIndex])+" in gameRow "+gameRowString)
      if (team2Object == None):
         print("WARNING: skipping game because of unknown Team2="+str(teamSeasons["Team2"][gameIndex])+" in gameRow "+gameRowString)
      if (team1Object != None and team2Object != None and team1Object.GetConferenceName(year) != None and team2Object.GetConferenceName(year) != None):
         # Only add a game if both teams are active for that season, determined by if they have a conference defined.
         if (roundsByYearDict[year].get(roundNumber) == None):
            # Add new round.  
            roundsByYearDict[year][roundNumber] = roundName
            #print("::: adding roundName "+roundName+" year "+str(year)+" thisGameSeason.seasonYear "+str(thisGameSeason.seasonYear))
            thisGameSeason.roundNames.append(roundName)
            thisGameSeason.seasonInfo[roundName] = [] # array containing Game objects
            thisGameSeason.roundIndex[roundName] = roundNumber
            thisGameSeason.roundCount += 1
         # Start out by initializing with average power per division.  This will be overridden to previous seasons power when team continues from previous year.
         avePower1 = gameSeason.GameSeason.GetDefaultPower(team1Object.GetDivisionName(year))
         team1Object.SetPower(year, avePower1["offense"], avePower1["defense"])
         avePower2 = gameSeason.GameSeason.GetDefaultPower(team2Object.GetDivisionName(year))
         team2Object.SetPower(year, avePower2["offense"], avePower2["defense"])
         #print("::: adding game for year "+str(thisGameSeason.seasonYear)+" team1 "+team1Object.teamName+" conf "+team1Object.GetConferenceName(year)+" team2 "+team2Object.teamName+" conf "+team2Object.GetConferenceName(year)+" roundName "+roundName+" s1 "+str(teamSeasons["Score1"][gameIndex])+" s2 "+str(teamSeasons["Score2"][gameIndex]))
         thisGameSeason.AddGame(team1Object, team2Object, roundName, homeField, False, teamSeasons["Score1"][gameIndex], teamSeasons["Score2"][gameIndex])
         team1Object.SetOpponent(year, roundName, team2Object, homeField)
         team2Object.SetOpponent(year, roundName, team1Object, homeField-1.0)

   for keyIndex in sorted(gameSeasonDict.keys()):
      thisGameSeason = gameSeasonDict[keyIndex]
      if (thisGameSeason.seasonYear == currentSeason-1 or thisGameSeason.seasonYear == currentSeason):
         gameSeasons.append(gameSeasonDict[keyIndex])

   if (verbose):
      print("SEASONS:")
      print("")
      for thisGameSeason in gameSeasons:
         thisGameSeason.PrintSeason()
         print("")
         print("")
 
else:

   print("::: createSimulatedSeason")
   uniqueTeamNames = True
   for count in range(2):
      thisGameSeason = gameSeason.GameSeason(currentSeason, gameSimulator, "gs"+str(count))
      thisGameSeason.CreateSimulatedSeason("NCAA2Divisions", True, initializePowerFromPrevious, uniqueTeamNames)
      gameSeasons.append(thisGameSeason)
      if (verbose):
         print("::: gameSeason "+thisGameSeason.name+" seasonYear "+str(thisGameSeason.seasonYear))
         thisGameSeason.PrintSeason()
         print("::: by schedules")
         thisGameSeason.PrintTeamSchedules()
         print("")
         print("")

#
# 
#

random.seed(fitRandomSeed)

elementsToFit = []
elementsToFit.append({'teamPowers': fitTeamPowers, 'linearParameters': False, 'homeFieldAdvantage': False, 'deprioritizeTotalForCost': deprioritizeTotalForCost})

if (fitTeamPowers):
   fitResults = []
   #for thisGameSeason in gameSeasons:
   for iSeason in range(2):
      thisGameSeason = gameSeasons[iSeason]
      gameSimulator.adamParams["beta1t"] = 1.0
      gameSimulator.adamParams["beta2t"] = 1.0
      previousTotalCount = -1
      fitResults.append({})
      roundCount = thisGameSeason.roundCount
      if (iSeason > 0):
         roundCount = maxRound
      for iRound in range(roundCount):
         costInfo = {}
         reachedStoppingCost = False
         if (iRound == maxRound - 1 and iSeason > 0):
            # Don't run fit for prediction week. Note that prediction week will be thisGameSeason.roundCount+1 for getting the results for final round of the season.  
            # TODO: may need additional logic in other sections below for when maxRound is out of bounds for the seasons' rounds, i.e. when getting results for final round of the season.
            reachedStoppingCost = True
         for iPass in range(numberOfIterationsForEachWeek):
            if (not reachedStoppingCost):
               #print("gameSeason "+thisGameSeason.name+" season "+str(thisGameSeason.seasonYear)+" round "+thisGameSeason.roundNames[iRound]+ " iPass "+str(iPass))
               for iFit in range(len(elementsToFit)):
                  costInfo = thisGameSeason.AdjustPowerAndLinearParametersFromScores(
                     learningRate,
                     adamUpdateEnable,
                     elementsToFit[iFit],
                     None,
                     0.001,
                     verbose, # verbose
                     False,
                     None, #forceAverageOffense,
                     None, #forceAverageDefense,
                     iRound+1)
                  # Adjust learning rate
                  costReductionPercentage = 100*(costInfo["costOrig"]-costInfo["costUpdated"])/max(1,costInfo["costOrig"])
                  if (iPass == 0 or iPass == numberOfIterationsForEachWeek-1 or iPass % 10 == 0):
                     print("::: costInfo for "+thisGameSeason.name+" "+str(thisGameSeason.seasonYear)+" round "+thisGameSeason.roundNames[iRound]+" pass "+str(iPass)+" cost="+str(costInfo["costUpdated"])+" costReduction="+str(costReductionPercentage)+"% gameCount="+str(costInfo["count"]))
                  #if (costReductionPercentage < 0.0001 or costInfo["costUpdated"] == 0):
                  if (costInfo["costOrig"] == 0 or costInfo["count"] == previousTotalCount):
                     reachedStoppingCost = True
         if (not reachedStoppingCost):
            previousTotalCount = costInfo["count"] 
         # Store the fit results by team.
         if (thisGameSeason.seasonInfo.get(thisGameSeason.roundNames[iRound]) != None):
            for game in thisGameSeason.seasonInfo[thisGameSeason.roundNames[iRound]]:
               if (fitResults[iSeason].get(game.team1Object.teamId) == None):
                  fitResults[iSeason][game.team1Object.teamId] = [] # array of games for each team
               team1Results = {}
               power1 = game.team1Object.GetPower(thisGameSeason.seasonYear)
               team1Results["roundName"] = thisGameSeason.roundNames[iRound]
               team1Results["roundIndex"] = iRound
               team1Results["defenseFit"] = power1["defense"]
               team1Results["offenseFit"] = power1["offense"]
               # TODO: add additional information to use in making forecast and rankings.
               fitResults[iSeason][game.team1Object.teamId].append(team1Results)
               # The "actual" power will be the fit power for the entire season.  Will go ahead and fill it in
               # here, updating it for each round, leaving the final update to the the "actual"
               game.team1Object.SetPowerActual(thisGameSeason.seasonYear, power1["offense"], power1["defense"])
               game.team1Object.SetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRound], power1["offense"], power1["defense"])
               if (fitResults[iSeason].get(game.team2Object.teamId) == None):
                  fitResults[iSeason][game.team2Object.teamId] = [] # array of games for each team
               team2Results = {}
               power2 = game.team2Object.GetPower(thisGameSeason.seasonYear)
               team2Results["roundName"] = thisGameSeason.roundNames[iRound]
               team2Results["roundIndex"] = iRound
               team2Results["defenseFit"] = power2["defense"]
               team2Results["offenseFit"] = power2["offense"]
               fitResults[iSeason][game.team2Object.teamId].append(team2Results)
               game.team2Object.SetPowerActual(thisGameSeason.seasonYear, power2["offense"], power2["defense"])
               game.team2Object.SetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRound], power2["offense"], power2["defense"])
      # Populate next season's initial power with the current season's final fit power.
      if (useTeamListFile != None and useTeamSeasonsFile != None and iSeason > 0 and iSeason < len(gameSeasons)):
         activeTeams = team.Team.GetListOfTeamIds(thisGameSeason.seasonYear)
         for teamId in activeTeams:
            teamObject = team.Team.teamObjectById[teamId]
            power = teamObject.GetPowerActual(thisGameSeason.seasonYear)
            teamObject.SetPower(thisGameSeason.seasonYear+1, power["offense"], power["defense"])

   # tmp - print out some results
   #print("::: team-25 fit powers: "+str(fitResults[iSeason][25]))

def GetCurrentPower(teamObject,iRound,thisGameSeason,gameCountByTeamId,currentPowerByTeamIdAndRound):
#  teamId = teamObject.teamId
#  if (gameCountByTeamId.get(teamId) == None):
#     gameCountByTeamId[teamId] = {}
#     gameCountByTeamId[teamId]["gameCount"] = 0
#     gameCountByTeamId[teamId]["countByRoundName"] = {}
#  power = {}
#  if (currentPowerByTeamIdAndRound.get(teamId) == None):
#     currentPowerByTeamIdAndRound[teamId] = {}
#  if (currentPowerByTeamIdAndRound[teamId].get(iRound) == None):
#     prevSeasonPower = teamObject.GetPowerActual(thisGameSeason.seasonYear-1)
#     if (prevSeasonPower["offense"] == None):
#        prevSeasonPower = gameSeason.GameSeason.GetDefaultPower(teamObject.GetDivisionName(year))
#     iRoundPrev = iRound - 1
#     while (iRoundPrev >= 0 and teamObject.GetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRoundPrev]) == None):
#        iRoundPrev -= 1
#     if (teamId == 24):
#        print("::: GetCurrentPower teamId=24 iRound "+str(iRound)+" gameCount "+str(gameCountByTeamId[teamId]["gameCount"])+" iRoundPrev "+str(iRoundPrev))
#     if (gameCountByTeamId[teamId]["gameCount"]-1 < 0 or iRoundPrev < 0):
#        # First game, so use the previous season's power.
#        useDef = prevSeasonPower["defense"]
#        useOff = prevSeasonPower["offense"]
#        if (teamId == 24):
#           print("::: GetCurrentPower teamId=24 prevSeasonPower "+str(prevSeasonPower))
#     else:
#        # Use the fit power from the previous round.
#        fitPower = teamObject.GetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRoundPrev])
#        weight = gameSimulator.weekWeights[gameCountByTeamId[teamId]["gameCount"]-1]
#        useDef = weight*fitPower["defense"] + (1.-weight)*prevSeasonPower["defense"]
#        useOff = weight*fitPower["offense"] + (1.-weight)*prevSeasonPower["offense"]
#        if (teamId == 24):
#           print("::: GetCurrentPower teamId=24 prevSeasonPower "+str(prevSeasonPower)+" weight "+str(weight)+" fitPower "+str(fitPower)+" off "+str(useOff)+" def "+str(useDef))
#     power["offense"] = useOff
#     power["defense"] = useDef
#     currentPowerByTeamIdAndRound[teamId][iRound] = power
#  else:
#     power = currentPowerByTeamIdAndRound[teamId][iRound]
#  if (teamId == 24):
#     print("::: GetCurrentRank teamId=24 precomputed power "+str(power)) 
#  return power
   teamId = teamObject.teamId
   if (gameCountByTeamId.get(teamId) == None):
      gameCountByTeamId[teamId] = {}
      gameCountByTeamId[teamId]["gameCount"] = 0
      gameCountByTeamId[teamId]["countByRoundName"] = {}
      gameCountByTeamId[teamId]["countByRoundNumber"] = {}
   power = {}
   if (currentPowerByTeamIdAndRound.get(teamId) == None):
      currentPowerByTeamIdAndRound[teamId] = {}

   if (currentPowerByTeamIdAndRound[teamId].get(iRound) == None):
      prevSeasonPower = teamObject.GetPowerActual(thisGameSeason.seasonYear-1)
      seasonChange = gameSeason.GameSeason.GetAveSeasonChange(teamObject.GetDivisionName(year))
      if (prevSeasonPower["offense"] == None):
         prevSeasonPower = gameSeason.GameSeason.GetDefaultPower(teamObject.GetDivisionName(year))

      # Find the newest round with a power.  Note that this can be a week in which the team did not play but for which a power was set.
      iRoundPrev = iRound - 1
      while (iRoundPrev >= 0 and teamObject.GetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRoundPrev]) == None):
      #while (iRoundPrev >= 0 and gameCountByTeamId[teamId]["countByRoundNumber"].get(iRoundPrev) == None):
         iRoundPrev -= 1

      if (iRoundPrev < 0 or gameCountByTeamId[teamId]["countByRoundNumber"].get(iRoundPrev) == None or gameCountByTeamId[teamId]["countByRoundNumber"][iRoundPrev] <= 0):
         # First game, so use the previous season's power.
         useDef = prevSeasonPower["defense"]
         useOff = prevSeasonPower["offense"]
      else:
         # Use the fit power from the previous round.
         fitPower = teamObject.GetFitPower(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRoundPrev])
         weight = gameSimulator.weekWeights[gameCountByTeamId[teamId]["countByRoundNumber"][iRoundPrev]]
         useDef = weight*fitPower["defense"] + (1.-weight)*prevSeasonPower["defense"]
         useOff = weight*fitPower["offense"] + (1.-weight)*prevSeasonPower["offense"]
      power["offense"] = useOff
      power["defense"] = useDef

      # Get the change in power between last two games.
      # Find the last two rounds with actual games.
      iRoundGame1 = iRound - 1
      while (iRoundGame1 >= 0 and teamObject.GetOpponent(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRoundGame1]) == None):
         iRoundGame1 -= 1
      if (iRoundGame1 < 0):
         power["offenseUncertainty"] = seasonChange["offense"]
         power["defenseUncertainty"] = seasonChange["defense"]
      else:
         iRoundGame2 = iRoundGame1 - 1
         while (iRoundGame2 >= 0 and teamObject.GetOpponent(thisGameSeason.seasonYear, thisGameSeason.roundNames[iRoundGame2]) == None):
            iRoundGame2 -= 1
         if (iRoundGame2 < 0):
            useDefPrev = prevSeasonPower["defense"]
            useOffPrev = prevSeasonPower["offense"]
         else:
            weightPrev = gameSimulator.weekWeights[gameCountByTeamId[teamId]["countByRoundNumber"][iRoundGame2]]
            useDefPrev = weightPrev*fitPower["defense"] + (1.-weightPrev)*prevSeasonPower["defense"]
            useOffPrev = weightPrev*fitPower["offense"] + (1.-weightPrev)*prevSeasonPower["offense"]
         power["offenseUncertainty"] = abs(useOff - useOffPrev)
         power["defenseUncertainty"] = abs(useDef - useDefPrev)

      currentPowerByTeamIdAndRound[teamId][iRound] = power
   else:
      power = currentPowerByTeamIdAndRound[teamId][iRound]
   return power

gameCountByTeamId = {}
currentPowerByTeamIdAndRound = {}
winScoreByTeamId = {}
teamIdByWinScore = {}
earnedRankByTeamId = {}
teamIdByEarnedScore = {}

numProbBins = 5
forecastStatistics = {}
predictions = {}
results = {}
#for thisGameSeason in gameSeasons:
iPredOffset = 0
if (minimalFirstRound):
   iPredOffset = -1
iSeason = 1
thisGameSeason = gameSeasons[iSeason]

for iProbBin in range(numProbBins+1):
   forecastStatistics[iProbBin] = {}
   forecastStatistics[iProbBin]["expected"] = 0
   forecastStatistics[iProbBin]["actual"] = 0
   forecastStatistics[iProbBin]["total"] = 0
roundCount = thisGameSeason.roundCount

outputFileObject = None
if (outputFile != None):
   outputFileObject = open(outputFile, "w")

forecastStatisticsByRound = {}

if (iSeason > 0):
   roundCount = maxRound
for iRound in range(roundCount):
   forecastStatisticsByRound[thisGameSeason.roundNames[iRound]] = {}
   for gameType in ["conf", "nonConf", "nonDiv"]:
      forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType] = {}
      for iProbBin in range(numProbBins+1):
         forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType][iProbBin] = {}
         forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType][iProbBin]["expected"] = 0
         forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType][iProbBin]["actual"] = 0
         forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType][iProbBin]["total"] = 0
   if (thisGameSeason.seasonInfo.get(thisGameSeason.roundNames[iRound]) != None and iRound <= roundCount -2):
      for game in thisGameSeason.seasonInfo[thisGameSeason.roundNames[iRound]]:
         # Set game counts
         if (gameCountByTeamId.get(game.team1Object.teamId) == None):
            gameCountByTeamId[game.team1Object.teamId] = {}
            gameCountByTeamId[game.team1Object.teamId]["gameCount"] = 0
            gameCountByTeamId[game.team1Object.teamId]["countByRoundName"] = {}
            gameCountByTeamId[game.team1Object.teamId]["countByRoundNumber"] = {}
         gameCountByTeamId[game.team1Object.teamId]["gameCount"] += 1
         gameCountByTeamId[game.team1Object.teamId]["countByRoundName"][thisGameSeason.roundNames[iRound]] = gameCountByTeamId[game.team1Object.teamId]["gameCount"]
         gameCountByTeamId[game.team1Object.teamId]["countByRoundNumber"][iRound] = gameCountByTeamId[game.team1Object.teamId]["gameCount"]
         if (gameCountByTeamId.get(game.team2Object.teamId) == None):
            gameCountByTeamId[game.team2Object.teamId] = {}
            gameCountByTeamId[game.team2Object.teamId]["gameCount"] = 0
            gameCountByTeamId[game.team2Object.teamId]["countByRoundName"] = {}
            gameCountByTeamId[game.team2Object.teamId]["countByRoundNumber"] = {}
         gameCountByTeamId[game.team2Object.teamId]["gameCount"] += 1
         gameCountByTeamId[game.team2Object.teamId]["countByRoundName"][thisGameSeason.roundNames[iRound]] = gameCountByTeamId[game.team2Object.teamId]["gameCount"]
         gameCountByTeamId[game.team2Object.teamId]["countByRoundNumber"][iRound] = gameCountByTeamId[game.team2Object.teamId]["gameCount"]

   if (iRound == roundCount - 2):
      # Compute the current power -- this is the fit power of the last week with scores then averaged with a weight to the previous season's power.
      for teamId in team.Team.GetListOfTeamIds(thisGameSeason.seasonYear):
         teamObject = team.Team.teamObjectById[teamId]
         GetCurrentPower(teamObject,iRound,thisGameSeason,gameCountByTeamId,currentPowerByTeamIdAndRound)

      # Compute rankings
      numTeams = len(team.Team.GetListOfTeamIds(thisGameSeason.seasonYear))
      #teamCounts = {}
      #teamCounts["total"] = 0
      # Power rankings
      for teamId1 in team.Team.GetListOfTeamIds(thisGameSeason.seasonYear):
         team1Object = team.Team.teamObjectById[teamId1]
         #if (teamCounts.get(team1Object.GetDivisionName(thisGameSeason.seasonYear)) == None):
         #   teamCounts[team1Object.GetDivisionName(thisGameSeason.seasonYear)] = 0
         #teamCounts[team1Object.GetDivisionName(thisGameSeason.seasonYear)] += 1
         #teamCounts["total"] += 1
         power1 = GetCurrentPower(team1Object,iRound,thisGameSeason,gameCountByTeamId,currentPowerByTeamIdAndRound)
         if (team1Object.teamId == 24):
            print("::: power rank teamId=24 rank power "+str(power1)) 
         for teamId2 in team.Team.GetListOfTeamIds(thisGameSeason.seasonYear):
            if (teamId1 < teamId2):
               team2Object = team.Team.teamObjectById[teamId2]
               power2 = GetCurrentPower(team2Object,iRound,thisGameSeason,gameCountByTeamId,currentPowerByTeamIdAndRound)
               scoreAndProb = gameSimulator.GetScoreAndProbability(power1["offense"], power1["defense"], power2["offense"], power2["defense"], 0.5)
               if (winScoreByTeamId.get(teamId1) == None):
                  winScoreByTeamId[teamId1] = {}
                  winScoreByTeamId[teamId1]["winCount"] = 0
                  winScoreByTeamId[teamId1]["winSpread"] = 0.
               if (winScoreByTeamId.get(teamId2) == None):
                  winScoreByTeamId[teamId2] = {}
                  winScoreByTeamId[teamId2]["winCount"] = 0
                  winScoreByTeamId[teamId2]["winSpread"] = 0.
               winScoreByTeamId[teamId1]["winSpread"] += scoreAndProb["score1"] - scoreAndProb["score2"]
               winScoreByTeamId[teamId2]["winSpread"] += scoreAndProb["score2"] - scoreAndProb["score1"]
               if (scoreAndProb["probability"] > 0.5):
                  winScoreByTeamId[teamId1]["winCount"] += 1
               elif (scoreAndProb["probability"] < 0.5):
                  winScoreByTeamId[teamId2]["winCount"] += 1
               else:
                  winScoreByTeamId[teamId1]["winCount"] += 0.5
                  winScoreByTeamId[teamId2]["winCount"] += 0.5
      for teamId in team.Team.GetListOfTeamIds(thisGameSeason.seasonYear):
         winScoreByTeamId[teamId]["winSpread"] /= 100*(numTeams-1)
         winScore = winScoreByTeamId[teamId]["winCount"] + winScoreByTeamId[teamId]["winSpread"]
         winScore = int(10000*winScore+0.5)
         if (teamIdByWinScore.get(winScore) == None):
            teamIdByWinScore[winScore] = []
         teamIdByWinScore[winScore].append(teamId)
      powerRankings = {}
      powerRankings["overall"] = []
      sortedWinScores = sorted(teamIdByWinScore.keys())
      sortedWinScores.reverse()
      for winScore in sortedWinScores:
         for teamId in teamIdByWinScore[winScore]:
            teamObject = team.Team.teamObjectById[teamId]
            if (powerRankings.get(teamObject.GetDivisionName(thisGameSeason.seasonYear)) == None):
               powerRankings[teamObject.GetDivisionName(thisGameSeason.seasonYear)] = []
            powerRankings[teamObject.GetDivisionName(thisGameSeason.seasonYear)].append(teamId)
            powerRankings["overall"].append(teamId)
      for rankKey in sorted(powerRankings.keys()):
         print("::: rankings for "+rankKey)
         outString = "::: rankings for "+rankKey
         print(outString)
         if (outputFileObject != None):
            outputFileObject.write(outString+"\n")
         for iRank in range(len(powerRankings[rankKey])):
            teamId = powerRankings[rankKey][iRank]
            teamObject = team.Team.teamObjectById[teamId]
            outString = str(iRank+1)+" "+teamObject.teamName
            print(outString)
            if (outputFileObject != None):
               outputFileObject.write(outString+"\n")

      # Earned rankings
      for teamId in team.Team.GetListOfTeamIds(thisGameSeason.seasonYear):
         team1Object = team.Team.teamObjectById[teamId]
         earnedRankByTeamId[teamId] = {}
         earnedRankByTeamId[teamId]["winCountExpected"] = 0
         earnedRankByTeamId[teamId]["winCountActual"] = 0
         earnedRankByTeamId[teamId]["winCountExpectedOther"] = 0
      for iRoundEarned in range(iRound+1): # range over 0 to iRound, where iRound=roundCount-2
         if (thisGameSeason.seasonInfo.get(thisGameSeason.roundNames[iRoundEarned]) != None):
            for game in thisGameSeason.seasonInfo[thisGameSeason.roundNames[iRoundEarned]]:
               power1 = GetCurrentPower(game.team1Object,iRound,thisGameSeason,gameCountByTeamId,currentPowerByTeamIdAndRound)
               power2 = GetCurrentPower(game.team2Object,iRound,thisGameSeason,gameCountByTeamId,currentPowerByTeamIdAndRound)
               scoreAndProb = gameSimulator.GetScoreAndProbability(power1["offense"], power1["defense"], power2["offense"], power2["defense"], game.homeField)
               earnedRankByTeamId[game.team1Object.teamId]["winCountExpected"] += scoreAndProb["probability"]
               earnedRankByTeamId[game.team2Object.teamId]["winCountExpected"] += 1. - scoreAndProb["probability"]
               if (game.score1 > game.score2):
                  earnedRankByTeamId[game.team1Object.teamId]["winCountActual"] += 1
               elif (game.score2 > game.score1):
                  earnedRankByTeamId[game.team2Object.teamId]["winCountActual"] += 1
               else:
                  earnedRankByTeamId[game.team1Object.teamId]["winCountActual"] += 0.5
                  earnedRankByTeamId[game.team2Object.teamId]["winCountActual"] += 0.5
               for teamIdOther in team.Team.GetListOfTeamIds(thisGameSeason.seasonYear):
                  teamObjectOther = team.Team.teamObjectById[teamIdOther]
                  powerOther = GetCurrentPower(teamObjectOther,iRound,thisGameSeason,gameCountByTeamId,currentPowerByTeamIdAndRound)
                  scoreAndProb1Other = gameSimulator.GetScoreAndProbability(powerOther["offense"], powerOther["defense"], power2["offense"], power2["defense"], game.homeField)
                  scoreAndProb2Other = gameSimulator.GetScoreAndProbability(power1["offense"], power1["defense"], powerOther["offense"], powerOther["defense"], game.homeField)
                  if (teamIdOther != game.team1Object.teamId):
                     earnedRankByTeamId[game.team1Object.teamId]["winCountExpectedOther"] += scoreAndProb1Other["probability"]
                  if (teamIdOther != game.team2Object.teamId):
                     earnedRankByTeamId[game.team2Object.teamId]["winCountExpectedOther"] += 1. - scoreAndProb2Other["probability"]
      for teamId in team.Team.GetListOfTeamIds(thisGameSeason.seasonYear):
         earnedRankByTeamId[teamId]["winCountExpectedOther"] /= (numTeams-1)
         earnedRank = earnedRankByTeamId[teamId]["winCountActual"] - earnedRankByTeamId[teamId]["winCountExpectedOther"]
         earnedScore = int(10000*(earnedRank+thisGameSeason.roundCount)+0.5)
         if (teamIdByEarnedScore.get(earnedScore) == None):
            teamIdByEarnedScore[earnedScore] = []
         teamIdByEarnedScore[earnedScore].append(teamId)
      earnedRankings = {}
      earnedRankings["overall"] = []
      sortedEarnedScores = sorted(teamIdByEarnedScore.keys())
      sortedEarnedScores.reverse()
      for earnedScore in sortedEarnedScores:
         for teamId in teamIdByEarnedScore[earnedScore]:
            teamObject = team.Team.teamObjectById[teamId]
            if (earnedRankings.get(teamObject.GetDivisionName(thisGameSeason.seasonYear)) == None):
               earnedRankings[teamObject.GetDivisionName(thisGameSeason.seasonYear)] = []
            earnedRankings[teamObject.GetDivisionName(thisGameSeason.seasonYear)].append(teamId)
            earnedRankings["overall"].append(teamId)
      for rankKey in sorted(earnedRankings.keys()):
         outString = ("::: earned rankings for "+rankKey)
         print(outString)
         if (outputFileObject != None):
            outputFileObject.write(outString+"\n")
         for iRank in range(len(earnedRankings[rankKey])):
            teamId = earnedRankings[rankKey][iRank]
            teamObject = team.Team.teamObjectById[teamId]
            outString = (str(iRank+1)+" "+teamObject.teamName
               +" actualWins="+str(earnedRankByTeamId[teamId]["winCountActual"])
               +" expectedWins="+str(earnedRankByTeamId[teamId]["winCountExpected"])
               +" otherWins="+str(earnedRankByTeamId[teamId]["winCountExpectedOther"])
               +" actual-other="+str(earnedRankByTeamId[teamId]["winCountActual"]-earnedRankByTeamId[teamId]["winCountExpectedOther"]))
            print(outString)
            if (outputFileObject != None):
               outputFileObject.write(outString+"\n")

   if (thisGameSeason.seasonInfo.get(thisGameSeason.roundNames[iRound]) != None):
      # For all rounds up through 
      for game in thisGameSeason.seasonInfo[thisGameSeason.roundNames[iRound]]:
         # Get current powers for team1 & team2
         power1 = GetCurrentPower(game.team1Object,iRound,thisGameSeason,gameCountByTeamId,currentPowerByTeamIdAndRound)
         useDef1 = power1["defense"]
         useOff1 = power1["offense"]
         power2 = GetCurrentPower(game.team2Object,iRound,thisGameSeason,gameCountByTeamId,currentPowerByTeamIdAndRound)
         useDef2 = power2["defense"]
         useOff2 = power2["offense"]
         #print(":::-1 game "+game.team1Object.teamName+" "+game.team2Object.teamName+" o1Del="+str(power1["offenseUncertainty"])+" d1Del="+str(power1["defenseUncertainty"])+" o2Del="+str(power2["offenseUncertainty"])+" d2Del="+str(power2["defenseUncertainty"]))
         if (usePowerUncertainty):
            scoreAndProb = gameSimulator.GetScoreAndProbability(useOff1, useDef1, useOff2, useDef2, game.homeField, power1["offenseUncertainty"], power1["defenseUncertainty"], power2["offenseUncertainty"], power2["defenseUncertainty"])
         else:
            scoreAndProb = gameSimulator.GetScoreAndProbability(useOff1, useDef1, useOff2, useDef2, game.homeField)
         if (iRound <= roundCount-2):
            # Results and rankings
            probability = scoreAndProb["probability"]
            if (probability < 0.5):
               probability = 1. - probability
            iProbBin = int(2*numProbBins*(probability-0.5))
            if (game.team1Object.GetDivisionName(thisGameSeason.seasonYear) != game.team2Object.GetDivisionName(thisGameSeason.seasonYear)):
               gameType = "nonDiv"
            elif (game.team1Object.GetConferenceName(thisGameSeason.seasonYear) != game.team2Object.GetConferenceName(thisGameSeason.seasonYear)):
               gameType = "nonConf"
            else:
               gameType = "conf"
            forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType][iProbBin]["expected"] += probability
            forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType][iProbBin]["total"] += 1
            forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType][numProbBins]["expected"] += probability
            forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType][numProbBins]["total"] += 1
            forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType][numProbBins]["year"] = thisGameSeason.seasonYear
            if (iRound == roundCount-2):
               forecastStatistics[iProbBin]["expected"] += probability
               forecastStatistics[iProbBin]["total"] += 1
               # last bin, numProbBins, is for total across all games
               forecastStatistics[numProbBins]["expected"] += probability
               forecastStatistics[numProbBins]["total"] += 1
               forecastStatistics[numProbBins]["year"] = thisGameSeason.seasonYear
               #print("::: spreadPred "+str(scoreAndProb["score1"]-scoreAndProb["score2"])+" spreadActual "+str(game.score1-game.score2))
            correctTeamPicked = False
            if ((scoreAndProb["score1"]-scoreAndProb["score2"]) * (game.score1-game.score2) > 0.):
               # result matches prediction
               forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType][iProbBin]["actual"] += 1
               forecastStatisticsByRound[thisGameSeason.roundNames[iRound]][gameType][numProbBins]["actual"] += 1
               if (iRound == roundCount-2):
                  forecastStatistics[iProbBin]["actual"] += 1
                  forecastStatistics[numProbBins]["actual"] += 1
               correctTeamPicked = True
            # Previous round's result
            if (correctTeamPicked):
               resultProb = probability
            else:
               resultProb = 1. - probability
            if (iRound == roundCount-2):
               if (game.score1 >= game.score2):
                  outString = ("results: prob="+str(resultProb)+" "+game.team1Object.teamName+"--"+game.team2Object.teamName+" actual="+str(game.score1)+"--"+str(game.score2)+" predicted="+str(scoreAndProb["score1"])+"--"+str(scoreAndProb["score2"]))
                  print(outString)
                  if (outputFileObject != None):
                     outputFileObject.write(outString+"\n")
               else:
                  outString = ("results: prob="+str(resultProb)+" "+game.team2Object.teamName+"--"+game.team1Object.teamName+" actual="+str(game.score2)+"--"+str(game.score1)+" predicted="+str(scoreAndProb["score2"])+"--"+str(scoreAndProb["score1"]))
                  print(outString)
                  if (outputFileObject != None):
                     outputFileObject.write(outString+"\n")
         if (iRound == roundCount-1 and iRound <= thisGameSeason.roundCount-1):
            # Predictions
            if (scoreAndProb["score1"] >= scoreAndProb["score2"]):
               outString = ("predictions: prob="+str(scoreAndProb["probability"])+" "+game.team1Object.teamName+"--"+game.team2Object.teamName+" predicted="+str(scoreAndProb["score1"])+"--"+str(scoreAndProb["score2"])+" spread="+str(scoreAndProb["score1"]-scoreAndProb["score2"]))
               print(outString)
               if (outputFileObject != None):
                  outputFileObject.write(outString+"\n")
            else:
               outString = ("predictions: prob="+str(1.-scoreAndProb["probability"])+" "+game.team2Object.teamName+"--"+game.team1Object.teamName+"  predicted="+str(scoreAndProb["score2"])+"--"+str(scoreAndProb["score1"])+" spread="+str(scoreAndProb["score2"]-scoreAndProb["score1"]))
               print(outString)
               if (outputFileObject != None):
                  outputFileObject.write(outString+"\n")

print("::: forecastStatistics "+str(forecastStatistics))
print("::: detailedStatistics "+str(forecastStatisticsByRound))
if (outputFileObject != None):
   outputFileObject.write("::: forecastStatistics "+str(forecastStatistics))
   outputFileObject.write("::: detailedStatistics "+str(forecastStatisticsByRound))

if (saveTeamPowersToFile != None):
   teamNames = []
   teamIds = []
   year = []
   offense = []
   defense = []
   for teamName in team.Team.teamObjectByName:
      teamObject = team.Team.teamObjectByName[teamName]
      for thisGameSeason in gameSeasons:
         power = teamObject.GetPower(thisGameSeason.seasonYear)
         if (power["defense"] != None):
            teamNames.append(teamName)
            teamIds.append(teamObject.teamId)
            year.append(thisGameSeason.seasonYear)
            defense.append(power["defense"])
            offense.append(power["offense"])
   teamPowers = {'teamName': teamNames, 'teamId': teamIds, 'year': year, 'offense': offense, 'defense': defense}
   teamPowersPandas = pandas.DataFrame(teamPowers)
   teamPowersPandas.to_csv(saveTeamPowersToFile)


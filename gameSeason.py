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

import team
import game

#
# Game season
#

class GameSeason:
   """Class for tracking team matchups for one season"""
    
   divName = ["IA", "IAA", "NFC", "AFC"]

   nflOffAve = 0.5*(8.746010822715604+8.697896109582821)
   nflDefAve = 0.5*(6.647436890833853+6.786771620061824)
   nflOffRms = 0.5*(0.66504084828149+0.6742529837945488)
   nflDefRms = 0.5*(0.7263478336898975+0.7404993844482559)
   nflOffDelta = 0.5*(1.1238781993589093+1.0750144721553336)
   nflDefDelta = 0.5*(1.2652993994344026+1.3462043356492461)

   defAve = [7.417854504803192, 5.613700341866554, nflDefAve, nflDefAve]
   offAve = [10.264432962217683, 7.694884091447714, nflOffAve, nflOffAve]
   defDivRms = [1.2529964382387893, 1.2206797283294453, nflDefRms, nflDefRms]
   offDivRms = [1.1057786122920785, 1.341083431558368, nflOffRms, nflOffRms]

   defSeasonChange = [1.8424067896960112, 1.955, nflDefDelta, nflDefDelta]
   offSeasonChange = [1.5770794762076497, 1.686645465207797, nflOffDelta, nflOffDelta] 

   def __init__(self, seasonYear, gameSimulator=None, name=""):
      self.seasonYear = seasonYear
      #self.roundNames # A round is a week or other interval over which results and predictions are made.
      self.roundNames = []
      self.seasonInfo = {}
      self.gameIndexByTeamAndRound = {}
      self.roundIndex = {}
      self.roundCount = 0
      self.teamSchedules = None
      self.gameSimulator = gameSimulator
      self.currentRound = 0
      self.name = name
      self.averageTeamPower= 0
      self.iDivOffset = 0

   def GetDefaultPower(divisionName):
      iDiv = 0
      if (divisionName == GameSeason.divName[1]): # "IAA"
         iDiv = 1
      elif (divisionName == GameSeason.divName[2]): # NFL
         iDiv = 2
      elif (divisionName == GameSeason.divName[3]): # NFL
         iDiv = 3

      power = {}
      power["offense"] = GameSeason.offAve[iDiv]
      power["defense"] = GameSeason.defAve[iDiv]

      return power
      
   def CreateSimulatedSeason(self, seasonType="NCAA2Divsions", correlatePreviousSeasonsPower=True, initializePowerFromPrevious=False, uniqueTeamNames=False):

      year = self.seasonYear

      if (seasonType == "NCAA2Divisions"):

         self.iDivOffset = 0

         #
         # BFM NCAA derived statistics
         #

         # What does a typical season of NCAA Div 1 look like?
         #
         # Note: counts are teams, not games, i.e. for number of games, divide the below by 2.
         #
         # 2017 colI_2017wk00out totals: all  16   conf   0   nonConf  14   nonDiv   2  pre1
         # 2017 colI_2017wk01out totals: all 216   conf   8   nonConf 116   nonDiv  92  pre2
         # 2017 colI_2017wk02out totals: all 218   conf  24   nonConf 144   nonDiv  50  week1
         # 2017 colI_2017wk03out totals: all 220   conf  34   nonConf 158   nonDiv  28  week2
         # 2017 colI_2017wk04out totals: all 218   conf 142   nonConf  70   nonDiv   6  week3
         # 2017 colI_2017wk05out totals: all 214   conf 166   nonConf  46   nonDiv   2  week4  
         # 2017 colI_2017wk06out totals: all 222   conf 194   nonConf  26   nonDiv   2  week5
         # 2017 colI_2017wk07out totals: all 228   conf 212   nonConf  16   nonDiv   0  week6
         # 2017 colI_2017wk08out totals: all 220   conf 206   nonConf  14   nonDiv   0  week7
         # 2017 colI_2017wk09out totals: all 226   conf 216   nonConf   8   nonDiv   2  week8
         # 2017 colI_2017wk10out totals: all 242   conf 222   nonConf  20   nonDiv   0  week9
         # 2017 colI_2017wk11out totals: all 236   conf 222   nonConf  12   nonDiv   2  week10
         # 2017 colI_2017wk12out totals: all 240   conf 214   nonConf  16   nonDiv  10  week11
         # 2017 colI_2017wk13out totals: all 144   conf 116   nonConf  28   nonDiv   0  week12
         # 2017 colI_2017wk14out totals: all  50   conf  38   nonConf  12   nonDiv   0  week13
         # 2017 colI_2017wk15out totals: all  10   conf   0   nonConf  10   nonDiv   0  post1
         # 2017 colI_2017wk16out totals: all  82   conf   0   nonConf  82   nonDiv   0  post2
         # 2017 colI_2017wk17out totals: all   4   conf   2   nonConf   2   nonDiv   0  post3
         #
         # defAve, offAve - average for that divsion
         # defRms, offRms - variation of the def & off inside that division
         # defChange, offChange - RMS change for off & def between years
         #
         # Division IA  defAve  5.43 offAve  7.88 defRms 0.40 offRms 0.48 defChange 0.46 offChange 0.57
         # Division IAA defAve  4.88 offAve  6.97 defRms 0.41 offRms 0.48 defChange 0.47 offChange 0.54

         #defAve = [5.43, 4.88]
         #offAve = [7.88, 6.97]
         #defDivRms = [0.4, 0.41]
         #offDivRms = [0.48, 0.48]

         #defSeasonChange = [0.46, 0.47]
         #offSeasonChange = [0.57, 0.54]

         # 2023-07-30
         # found linear parameters with this (averaged over the 5 cases):
         #simFb-s20230727-f4-forceAve-adjustLin.out
         #spdCoefOff	spdCoefDef	totCoefOff	totCoefDef	totCoef0
         #7.45	4.7	2.83	-2.95	24.26
         # my $settings = "seasonRandomSeed=SEASONSEED fitRandomSeed=FITSEED numberOfSeasons=22 learningRateInitial=0.001 numberOfPassesThroughAllSeasons=2000 numberOfIterationsForEachSeason=10 roundScores=True verbose=True optimizeWeekWeights=False randomizeWeights=False randomizeTeamPowers=True randomizeLinearParameters=False correlatePreviousSeasonsPower=True minimizeChangePowerPerSeason=True fitTeamPowers=True fitLinearParameters=True fitHomeFieldAdvantage=False adamUpdateEnable=True deprioritizeTotalForCost=0.25 rmsMultiplier=1.0 constrainAveragePower=False forceAverageOffense=7.4 forceAverageDefense=5.2 spdCoefOffInit=16.5 spdCoefDefInit=20.5 totCoefOffInit=12 totCoefDefInit=-15.5 totCoef0Init=0.0 useTeamListFile=colI-list-1998_2019.csv useTeamSeasonsFile=colI-seasons-1998_2019.csv saveTeamPowersToFile=colI-fitPowers-1998_2019-s$seasonBaseSeed-fFITSEED-forceAve-adjustLin.csv";

         # python createAveragePowerStats.py
         #    Input Power Files: ['colI-fitPowers-1998_2019-s20230727-f0-noForce.csv', 'colI-fitPowers-1998_2019-s20230727-f1-noForce.csv', 'colI-fitPowers-1998_2019-s20230727-f2-noForce.csv', 'colI-fitPowers-1998_2019-s20230727-f3-noForce.csv', 'colI-fitPowers-1998_2019-s20230727-f4-noForce.csv']
         #    Division IAA offAve 7.455643920306612 defAve 5.4063563037702975 offRms 0.8016621507277709 defRms 0.8185274936094209 offDelta 1.689361193727034 defDelta 2.22529449771079
         #    Division IA offAve 10.242499686376325 defAve 7.684446393797453 offRms 0.6566279158077948 defRms 0.8343169692953073 offDelta 1.567377773036738 defDelta 2.0877683588910743
         #    colI-averagePowers-1998_2019-s20230727-noForce.csv

         #    Division IA offAve 
            # off def ave
            # 10.242499686376325 
            # 7.684446393797453 
            # off def RMS 
            # 0.6566279158077948 
            # 0.8343169692953073 
            # off def change
            # 1.567377773036738 
            # 2.0877683588910743
         #    Division IAA 
            # off def ave
            # 7.455643920306612 
            # 5.4063563037702975 
            # off def RMS 
            # 0.8016621507277709 
            # 0.8185274936094209 
            # off def change
            # 1.689361193727034 
            # 2.22529449771079

         # spdCoefOff	spdCoefDef	totCoefOff	totCoefDef	totCoef0
         # 8.08	5.93	2.76	-3.28	28.89
         # rerun fit with the derived linear parameters and without forcing off/def.
         #   my $settings = "seasonRandomSeed=SEASONSEED fitRandomSeed=FITSEED numberOfSeasons=22 learningRateInitial=0.001 numberOfPassesThroughAllSeasons=1600 numberOfIterationsForEachSeason=10 roundScores=True verbose=True optimizeWeekWeights=False randomizeWeights=False randomizeTeamPowers=True randomizeLinearParameters=False correlatePreviousSeasonsPower=True minimizeChangePowerPerSeason=True fitTeamPowers=True fitLinearParameters=False fitHomeFieldAdvantage=False adamUpdateEnable=True deprioritizeTotalForCost=0.25 rmsMultiplier=1.0 constrainAveragePower=False spdCoefOffInit=8.08 spdCoefDefInit=5.93 totCoefOffInit=2.76 totCoefDefInit=-3.28 totCoef0Init=28.89 useTeamListFile=colI-list-1998_2019.csv useTeamSeasonsFile=colI-seasons-1998_2019.csv saveTeamPowersToFile=colI-fitPowers-1998_2019-s$seasonBaseSeed-fFITSEED-noForce.csv";

         #   Input Power Files: ['colI-fitPowers-1998_2019-s20230805-f0-noForce.csv', 'colI-fitPowers-1998_2019-s20230805-f1-noForce.csv', 'colI-fitPowers-1998_2019-s20230805-f2-noForce.csv', 'colI-fitPowers-1998_2019-s20230805-f3-noForce.csv', 'colI-fitPowers-1998_2019-s20230805-f4-noForce.csv']
         # Division IAA offAve 7.694884091447714 defAve 5.613700341866554 offRms 1.341083431558368 defRms 1.2206797283294453 offDelta 1.686645465207797 defDelta 1.95521600195564
         # Division IA offAve 10.264432962217683 defAve 7.417854504803192 offRms 1.1057786122920785 defRms 1.2529964382387893 offDelta 1.5770794762076497 defDelta 1.8424067896960112

         # Division IAA 
            #offAve 7.694884091447714 
            #defAve 5.613700341866554 
            #offRms 1.341083431558368 
            #defRms 1.2206797283294453 
            #offDelta 1.686645465207797 
            #defDelta 1.95521600195564
         # Division IA 
            #offAve 10.264432962217683 
            #defAve 7.417854504803192 
            #offRms 1.1057786122920785 
            #defRms 1.2529964382387893 
            #offDelta 1.5770794762076497 
            #defDelta 1.8424067896960112

         numberTeamsInConference = 11
         numberConferencesInDivision = 10
         #numberTeamsInConference = 5
         #numberConferencesInDivision = 2

         self.roundNames = ["pre1", "pre2", 
                            "week1", "week2", "week3", "week4", "week5", "week6", "week7", 
                            "week8", "week9", "week10", "week11", "week12", "week13", 
                            "post1", "post2", "post3"]

         # Priority order for filling in conference games from sorting number of conference games in a given week.
         conferenceWeeksList = ["week9", "week10", "week8", "week11", "week6", "week7", "week5", 
                                "week4", "week3", "week12", "week13", "week2", "week1", "pre2"]

         nonConferenceWeekList = {}
         nonDivisionWeekList = {}

         nonConferenceWeekList["pre1"] = 14/2
         nonConferenceWeekList["pre2"] = 116/2
         nonConferenceWeekList["week1"] = 144/2
         nonConferenceWeekList["week2"] = 158/2
         nonConferenceWeekList["week3"] = 70/2
         nonConferenceWeekList["week4"] = 46/2
         nonConferenceWeekList["week5"] = 26/2
         nonConferenceWeekList["week6"] = 16/2
         nonConferenceWeekList["week7"] = 14/2
         nonConferenceWeekList["week8"] = 8/2
         nonConferenceWeekList["week9"] = 20/2
         nonConferenceWeekList["week10"] = 12/2
         nonConferenceWeekList["week11"] = 16/2
         nonConferenceWeekList["week12"] = 28/2
         nonConferenceWeekList["week13"] = 12/2

         nonDivisionWeekList["pre1"] = 2/2
         nonDivisionWeekList["pre2"] = 92/2
         nonDivisionWeekList["week1"] = 50/2
         nonDivisionWeekList["week2"] = 28/2
         nonDivisionWeekList["week3"] = 6/2
         nonDivisionWeekList["week4"] = 2/2
         nonDivisionWeekList["week5"] = 2/2
         nonDivisionWeekList["week6"] = 0
         nonDivisionWeekList["week7"] = 0
         nonDivisionWeekList["week8"] = 2/2
         nonDivisionWeekList["week9"] = 0
         nonDivisionWeekList["week10"] = 2/2
         nonDivisionWeekList["week11"] = 10/2
         nonDivisionWeekList["week12"] = 0
         nonDivisionWeekList["week13"] = 0

         divisionNames = ["D0", "D1"]

      elif (seasonType == "NFL"):

         self.iDivOffset = 2
         
         #
         # BFM NFL derived statistics
         #
         #week,conf,nonconf,nondiv
         #22wkb1,6,4,6
         #21wkb1,2,8,6
         #aveb1,4,6,6

         #22wk02,5,9,2
         #21wk02,6,6,4
         #ave02,6,7,3

         #22wk03,7,7,2
         #21wk03,4,6,6
         #ave03,5,7,4

         #22wk04,4,8,4
         #21wk04,4,8,4
         #ave04,4,8,4

         #22wk05,6,8,2
         #21wk05,5,7,4
         #ave05,6,7,3

         #22wk06,4,5,5
         #21wk06,3,6,5
         #ave06,3,6,5

         #22wk07,3,6,5
         #21wk07,2,7,4
         #ave07,2,7,5

         #22wk08,5,5,6
         #21wk08,5,6,4
         #ave08,5,5,5

         #22wk09,3,5,5
         #21wk09,3,3,8
         #ave09,3,4,6

         #22wk10,4,6,4
         #21wk10,4,6,4
         #ave10,4,6,4

         #22wk11,5,5,4
         #21wk11,4,6,5
         #ave11,5,5,4

         #22wk12,1,7,8
         #21wk12,5,5,5
         #ave12,3,6,7

         #22wk13,6,3,6
         #21wk13,7,3,4
         #ave13,6,3,5

         #22wk14,7,3,3
         #21wk14,7,7,0
         #ave14,7,5,2

         #22wk15,5,5,6
         #21wk15,8,6,2
         #ave15,7,5,4

         #22wk16,2,8,6
         #21wk16,6,6,4
         #ave16,4,7,5

         #22wk17,7,2,6
         #21wk17,5,7,4
         #ave17,6,5,5

         #22wk18,16,0,0
         #21wk18,16,0,0
         #ave18,16,0,0

         self.roundNames = ["pre1",
                            "week2", "week3", "week4", "week5", "week6", "week7", 
                            "week8", "week9", "week10", "week11", "week12", "week13", 
                            "week14", "week15", "week16", "week17", "week18",
                            "post1", "post2", "post3", "post4"]

         # Priority order for filling in conference games from sorting number of conference games in a given week.
         conferenceWeeksList = ["week18", "week15", "week14", "week2", "week17", "week5", "week13", "week3",
                                "week11", "week8", "week16", "week4", "week10", "pre1", "week9", "week12",
                                "week6", "week7"]

         nonConferenceWeekList = {}
         nonDivisionWeekList = {}

         nonConferenceWeekList["pre1"] = 6
         nonConferenceWeekList["week2"] = 7
         nonConferenceWeekList["week3"] = 7
         nonConferenceWeekList["week4"] = 8
         nonConferenceWeekList["week5"] = 7
         nonConferenceWeekList["week6"] = 6
         nonConferenceWeekList["week7"] = 7
         nonConferenceWeekList["week8"] = 5
         nonConferenceWeekList["week9"] = 4
         nonConferenceWeekList["week10"] = 6
         nonConferenceWeekList["week11"] = 5
         nonConferenceWeekList["week12"] = 6
         nonConferenceWeekList["week13"] = 3
         nonConferenceWeekList["week14"] = 5
         nonConferenceWeekList["week15"] = 5
         nonConferenceWeekList["week16"] = 7
         nonConferenceWeekList["week17"] = 5
         nonConferenceWeekList["week18"] = 0

         nonDivisionWeekList["pre1"] = 6
         nonDivisionWeekList["week2"] = 3
         nonDivisionWeekList["week3"] = 4
         nonDivisionWeekList["week4"] = 4
         nonDivisionWeekList["week5"] = 3
         nonDivisionWeekList["week6"] = 5
         nonDivisionWeekList["week7"] = 5
         nonDivisionWeekList["week8"] = 5
         nonDivisionWeekList["week9"] = 6
         nonDivisionWeekList["week10"] = 4
         nonDivisionWeekList["week11"] = 4
         nonDivisionWeekList["week12"] = 7
         nonDivisionWeekList["week13"] = 5
         nonDivisionWeekList["week14"] = 2
         nonDivisionWeekList["week15"] = 4
         nonDivisionWeekList["week16"] = 5
         nonDivisionWeekList["week17"] = 5
         nonDivisionWeekList["week18"] = 0

         divisionNames = ["D0", "D1"]

         numberTeamsInConference = 16
         numberConferencesInDivision = 4

         # fixme
         # python createAveragePowerStats.py
         #    Input Power Files: ['colI-fitPowers-1998_2019-s20230727-f0-noForce.csv', 'colI-fitPowers-1998_2019-s20230727-f1-noForce.csv', 'colI-fitPowers-1998_2019-s20230727-f2-noForce.csv', 'colI-fitPowers-1998_2019-s20230727-f3-noForce.csv', 'colI-fitPowers-1998_2019-s20230727-f4-noForce.csv']
         #    Division IAA offAve 7.455643920306612 defAve 5.4063563037702975 offRms 0.8016621507277709 defRms 0.8185274936094209 offDelta 1.689361193727034 defDelta 2.22529449771079
         #    Division IA offAve 10.242499686376325 defAve 7.684446393797453 offRms 0.6566279158077948 defRms 0.8343169692953073 offDelta 1.567377773036738 defDelta 2.0877683588910743
         #    colI-averagePowers-1998_2019-s20230727-noForce.csv

         #    Division IA offAve 
            # off def ave
            # 10.242499686376325 
            # 7.684446393797453 
            # off def RMS 
            # 0.6566279158077948 
            # 0.8343169692953073 
            # off def change
            # 1.567377773036738 
            # 2.0877683588910743
         #    Division IAA 
            # off def ave
            # 7.455643920306612 
            # 5.4063563037702975 
            # off def RMS 
            # 0.8016621507277709 
            # 0.8185274936094209 
            # off def change
            # 1.689361193727034 
            # 2.22529449771079

         # spdCoefOff	spdCoefDef	totCoefOff	totCoefDef	totCoef0
         # 8.08	5.93	2.76	-3.28	28.89
         # rerun fit with the derived linear parameters and without forcing off/def.
         #   my $settings = "seasonRandomSeed=SEASONSEED fitRandomSeed=FITSEED numberOfSeasons=22 learningRateInitial=0.001 numberOfPassesThroughAllSeasons=1600 numberOfIterationsForEachSeason=10 roundScores=True verbose=True optimizeWeekWeights=False randomizeWeights=False randomizeTeamPowers=True randomizeLinearParameters=False correlatePreviousSeasonsPower=True minimizeChangePowerPerSeason=True fitTeamPowers=True fitLinearParameters=False fitHomeFieldAdvantage=False adamUpdateEnable=True deprioritizeTotalForCost=0.25 rmsMultiplier=1.0 constrainAveragePower=False spdCoefOffInit=8.08 spdCoefDefInit=5.93 totCoefOffInit=2.76 totCoefDefInit=-3.28 totCoef0Init=28.89 useTeamListFile=colI-list-1998_2019.csv useTeamSeasonsFile=colI-seasons-1998_2019.csv saveTeamPowersToFile=colI-fitPowers-1998_2019-s$seasonBaseSeed-fFITSEED-noForce.csv";

         #   Input Power Files: ['colI-fitPowers-1998_2019-s20230805-f0-noForce.csv', 'colI-fitPowers-1998_2019-s20230805-f1-noForce.csv', 'colI-fitPowers-1998_2019-s20230805-f2-noForce.csv', 'colI-fitPowers-1998_2019-s20230805-f3-noForce.csv', 'colI-fitPowers-1998_2019-s20230805-f4-noForce.csv']
         # Division IAA offAve 7.694884091447714 defAve 5.613700341866554 offRms 1.341083431558368 defRms 1.2206797283294453 offDelta 1.686645465207797 defDelta 1.95521600195564
         # Division IA offAve 10.264432962217683 defAve 7.417854504803192 offRms 1.1057786122920785 defRms 1.2529964382387893 offDelta 1.5770794762076497 defDelta 1.8424067896960112

         # Division IAA 
            #offAve 7.694884091447714 
            #defAve 5.613700341866554 
            #offRms 1.341083431558368 
            #defRms 1.2206797283294453 
            #offDelta 1.686645465207797 
            #defDelta 1.95521600195564
         # Division IA 
            #offAve 10.264432962217683 
            #defAve 7.417854504803192 
            #offRms 1.1057786122920785 
            #defRms 1.2529964382387893 
            #offDelta 1.5770794762076497 
            #defDelta 1.8424067896960112

         #numberTeamsInConference = 5
         #numberConferencesInDivision = 2

      else: 
         print("GameSeason: ERROR: unsupported simulated season type "+seasonType)
         exit(1)

      teamNamesByDivision = {}
      teamNamesByConference = {}
      teamCount = 0

      for iDiv in range(len(divisionNames)):
         teamNamesByDivision[divisionNames[iDiv]] = {}
         for iConf in range(numberConferencesInDivision):
            conferenceName = "C"+str(iConf)+"."+divisionNames[iDiv]
            teamNamesByConference[conferenceName] = {}
            for iTeam in range(numberTeamsInConference):
               if (uniqueTeamNames):
                  teamName = "T"+str(iTeam).rjust(2,'0')+"."+conferenceName+"-"+self.name
               else:
                  teamName = "T"+str(iTeam).rjust(2,'0')+"."+conferenceName
               teamNamesByDivision[divisionNames[iDiv]][teamName] = 1
               teamNamesByConference[conferenceName][teamName] = 1
               teamObject = team.Team.teamObjectByName.get(teamName)
               firstSeason = False
               if (teamObject == None):
                  firstSeason = True
                  # Team has not yet been created.  Create it now.
                  #print("::: create team "+teamName+" for year "+str(year))
                  teamObject = team.Team(teamName, year, conferenceName, divisionNames[iDiv])
                  # Create power that is uncorreted from a previous year
                  defensePower = random.gauss(self.defAve[iDiv+self.iDivOffset], self.defDivRms[iDiv+self.iDivOffset])
                  offensePower = random.gauss(self.offAve[iDiv+self.iDivOffset], self.offDivRms[iDiv+self.iDivOffset])
                  if (correlatePreviousSeasonsPower):
                     # correlate the power to previous year.  note Previous=>Current
                     # set defensePrevious to previous years actual
                     defensePrevious = random.gauss(defensePower, self.defSeasonChange[iDiv+self.iDivOffset])
                     offensePrevious = random.gauss(offensePower, self.offSeasonChange[iDiv+self.iDivOffset])
                  else:
                     defensePrevious = random.gauss(self.defAve[iDiv+self.iDivOffset], self.defDivRms[iDiv+self.iDivOffset])
                     offensePrevious = random.gauss(self.offAve[iDiv+self.iDivOffset], self.offDivRms[iDiv+self.iDivOffset])
               else:
                  previousPower = teamObject.GetPowerActual(year-1)
                  defensePrevious = previousPower["defense"]
                  offensePrevious = previousPower["offense"]
                  # Use existing Team object
                  #print("::: use existing team "+teamName+" for year "+str(year))
                  teamObject.SetConferenceAndDivision(year, conferenceName, divisionNames[iDiv])
                  if (correlatePreviousSeasonsPower):
                     # correlate the power to previous year.
                     defensePower = random.gauss(defensePrevious, self.defSeasonChange[iDiv])
                     offensePower = random.gauss(offensePrevious, self.offSeasonChange[iDiv])
                  else:
                     defensePower = random.gauss(self.defAve[iDiv+self.iDivOffset], self.defDivRms[iDiv+self.iDivOffset])
                     offensePower = random.gauss(self.offAve[iDiv+self.iDivOffset], self.offDivRms[iDiv+self.iDivOffset])
                  if (defensePower < 0.0):
                     defensePower = 0.0
                  if (offensePower < 0.0):
                     offensePower = 0.0
                  teamObject.SetPower(year, offensePower, defensePower)
               teamCount += 1
               self.averageTeamPower += offensePower + defensePower*(-self.gameSimulator.totCoefOff/self.gameSimulator.totCoefDef)
               teamObject.SetPowerActual(year, offensePower, defensePower)
               if (firstSeason):
                  teamObject.SetPowerActual(year-1, offensePrevious, defensePrevious)
                  teamObject.SetPower(year-1, offensePrevious, defensePrevious)
               if (initializePowerFromPrevious):
                  teamObject.SetPower(year, offensePrevious, defensePrevious) # start of the season, the current power is last year's power
               else:
                  teamObject.SetPower(year, offensePower, defensePower)

      self.averageTeamPower /= teamCount

      for roundName in self.roundNames:
         self.seasonInfo[roundName] = [] # array containing Game objects
         self.roundIndex[roundName] = self.roundCount
         self.roundCount += 1

      # List of games for pre-season and regular season.  Enforces only one matchup between any two given teams for the regular season.
      roundNameForRegularSeasonMatchup = {} # Returns the round name for matchup of team1+team2 (games are added as team1+team2 as well as team2+team1).  Excludes tournaments (i.e. excludes repeat matchups in same season).

      # Conference games
      for conference in teamNamesByConference.keys():
         teamList1 = []
         for teamKey in teamNamesByConference[conference].keys():
            teamList1.append(str(teamKey))
          
         random.shuffle(teamList1)
         for team1 in teamList1:
            team1Object = team.Team.teamObjectByName.get(team1)
            homeCount = 0
            awayCount = 0
            teamList2 = []
            for teamKey in teamNamesByConference[conference].keys():
               teamList2.append(str(teamKey))
            random.shuffle(teamList2)
            for team2 in teamList2:
               if (team1 != team2):
                  team2Object = team.Team.teamObjectByName.get(team2)
                  roundName = roundNameForRegularSeasonMatchup.get(team1+"+"+team2)
                  if (roundName == None):
                     iRound = 0
                     foundWeek = False
                     while (iRound < len(conferenceWeeksList) and not foundWeek):
                        # Find first available week for matchup
                        if (team1Object.GetOpponent(year, conferenceWeeksList[iRound]) == None \
                                      and team2Object.GetOpponent(year, conferenceWeeksList[iRound]) == None):
                           if (awayCount >= homeCount):
                              self.AddGame(team1Object, team2Object, conferenceWeeksList[iRound], 1)
                              team1Object.SetOpponent(year, conferenceWeeksList[iRound], team2, 1)
                              team2Object.SetOpponent(year, conferenceWeeksList[iRound], team1, 0)
                              homeCount += 1
                           else:
                              self.AddGame(team2Object, team1Object, conferenceWeeksList[iRound], 1)
                              team1Object.SetOpponent(year, conferenceWeeksList[iRound], team2, 0)
                              team2Object.SetOpponent(year, conferenceWeeksList[iRound], team1, 1)
                              awayCount += 1
                           foundWeek = True
                           roundNameForRegularSeasonMatchup[team1+"+"+team2] = conferenceWeeksList[iRound]
                           roundNameForRegularSeasonMatchup[team2+"+"+team1] = conferenceWeeksList[iRound]
                        else:
                           iRound += 1  
                  else:
                     homeField = team1Object.GetHomeField(year, roundName)
                     if (homeField == 1):
                        homeCount +=1
                     elif (homeField == 0):
                        awayCount += 1    
                                                      
      # Fill in non-division & non-conference games

      for roundName in nonConferenceWeekList.keys():
         #year = gameSeason.seasonYear
         randomListOfTeamIds = team.Team.GetListOfTeamIds(year)
         random.shuffle(randomListOfTeamIds)
         numTeams = len(randomListOfTeamIds)
         nonConferenceCountRemaining = nonConferenceWeekList[roundName]
         nonDivisionCountRemaining = nonDivisionWeekList[roundName]
         idOffset1 = 0
         idOffset2 = 1
         while (nonConferenceCountRemaining > 0 or nonDivisionCountRemaining > 0):
            team1Object = team.Team.teamObjectById[randomListOfTeamIds[idOffset1]]
            team2Object = team.Team.teamObjectById[randomListOfTeamIds[(idOffset1+idOffset2) % numTeams]]
            matchupFound = False
            gotoNextTeam1 = False
            gotoNextTeam2 = False
            if (team1Object.GetOpponent(year, roundName) != None):
               # Team1 already has a game scheduled for this round.  Go to the next team1.
               gotoNextTeam1 = True
            elif (team2Object.GetOpponent(year, roundName) != None):
               # Team2 already has a game scheduled for this round.  Go to the next team2.
               gotoNextTeam2 = True
            else:
               # Team1 & team2 are both available.  See if they can fulfil the open matchups.
               sameDivision = team1Object.GetDivisionName(year) == team2Object.GetDivisionName(year)
               sameConference = team1Object.GetConferenceName(year) == team2Object.GetConferenceName(year)
               if (nonDivisionCountRemaining > 0 and not sameDivision):
                  matchupFound = True
                  nonDivisionCountRemaining -= 1
               elif (nonConferenceCountRemaining > 0 and sameDivision and not sameConference):
                  matchupFound = True
                  nonConferenceCountRemaining -= 1
               else:
                  gotoNextTeam2 = True 
            if (matchupFound):
               if (random.random() < 0.5):
                  self.AddGame(team1Object, team2Object, roundName, 1)
                  team1Object.SetOpponent(year, roundName, team2, 1)
                  team2Object.SetOpponent(year, roundName, team1, 0)
               else:
                  self.AddGame(team2Object, team1Object, roundName, 1)
                  team1Object.SetOpponent(year, roundName, team2, 0)
                  team2Object.SetOpponent(year, roundName, team1, 1)
               gotoNextTeam1 = True
            if (gotoNextTeam2):
               idOffset2 += 1
               if (idOffset2 >= numTeams):
                  # Have exhausted all possible team2s for this team1.  Go to next team1.
                  gotoNextTeam1 = True
            if (gotoNextTeam1):
               idOffset1 += 1
               idOffset2 = 1
               if (idOffset1 >= numTeams):
                  # Have exhausted all possible teams.  Quiting search for more games.
                  nonConferenceCountRemaining = 0
                  nonDivisionCountRemaining = 0

      # Post season

      # Will not do statistics for post-season at this point.

   def AddGame(self, team1, team2, roundName, homeField, generateScore=None, score1=None, score2=None):
      if (generateScore == None):
         # Default generateScore to True if gameSimulator has been provided.
         generateScore = not (self.gameSimulator == None)
      if (generateScore):
         actualPower1 = team1.GetPowerActual(self.seasonYear)
         actualPower2 = team2.GetPowerActual(self.seasonYear)
         scoreAndProbability = self.gameSimulator.GetRandomizedScore(
                actualPower1["offense"], actualPower1["defense"], 
                actualPower2["offense"], actualPower2["defense"], homeField)
         #for key in scoreAndProbability.keys():
         #    print("::: key " + key + "  = " + scoreAndProbability[key])
         #print("::: score1 "+str(scoreAndProbability["score1"]))
         score1 = scoreAndProbability["score1"]
         score2 = scoreAndProbability["score2"]
         if (self.gameSimulator.roundScores):
            score1 = int(score1+0.5)
            score2 = int(score2+0.5)
      self.seasonInfo[roundName].append(game.Game(team1, team2, roundName, homeField, score1, score2))
      gameIndex = len(self.seasonInfo[roundName]) - 1
      if (self.gameIndexByTeamAndRound.get(roundName) == None):
         self.gameIndexByTeamAndRound[roundName] = {}
      # Note that a team can only have one game per round.
      self.gameIndexByTeamAndRound[roundName][team1] = gameIndex
      self.gameIndexByTeamAndRound[roundName][team2] = gameIndex
        
   def SetTeamSchedules(self):
      self.teamSchedules = {}
      # Assemble an array of games for each team.
      for iRound in range(self.roundCount):
         if (self.seasonInfo.get(self.roundNames[iRound]) != None):
            for game in self.seasonInfo[self.roundNames[iRound]]:
               if (self.teamSchedules.get(game.team1Object.teamId) == None):
                  self.teamSchedules[game.team1Object.teamId] = []
               if (self.teamSchedules.get(game.team2Object.teamId) == None):
                  self.teamSchedules[game.team2Object.teamId] = []
               self.teamSchedules[game.team1Object.teamId].append(game.gameId)
               self.teamSchedules[game.team2Object.teamId].append(game.gameId)

   def PrintTeamSchedules(self):
      print("Schedule for season "+str(self.seasonYear)+" by teams")
      if (self.teamSchedules == None):
         self.SetTeamSchedules()
      for iTeam in sorted(self.teamSchedules.keys()):
         print("Team "+team.Team.teamObjectById[iTeam].teamName)
         for gameId in self.teamSchedules[iTeam]:
            game.Game.gameObjectById[gameId].PrintGame(game.Game.gameObjectById[gameId].roundName.ljust(6," ")+" ")

   def PrintSeason(self):
      print("Schedule for season "+str(self.seasonYear))
      countString = ""
      for iRound in range(self.roundCount):
         print("Round "+self.roundNames[iRound])
         conferenceCount = 0
         divisionCount = 0
         nondivisionCount = 0
         if (self.seasonInfo.get(self.roundNames[iRound]) != None):
            for game in self.seasonInfo[self.roundNames[iRound]]:
               game.PrintGame(self.roundNames[iRound].ljust(6," ")+" ")
               if (game.team1Object.GetConferenceName(self.seasonYear) == game.team2Object.GetConferenceName(self.seasonYear)):
                  conferenceCount += 1
               elif (game.team1Object.GetDivisionName(self.seasonYear) == game.team2Object.GetDivisionName(self.seasonYear)):
                  divisionCount += 1
               else:
                  nondivisionCount += 1
            countString += self.roundNames[iRound].ljust(6,' ')+" counts: conference="+str(conferenceCount).ljust(3,' ')+" division="+str(divisionCount).ljust(3,' ')+" nondivision="+str(nondivisionCount).ljust(3,' ')+"\n"
      print(countString, end='')

   def GetCurrentPower(self, teamObject, year, roundName):
      fitPower = teamObject.GetFitPower(year,roundName)
      while (fitPower == None):
         roundIndex = self.roundIndex[roundName]
#            print("::: GCP roundName "+roundName+" roundIndex "+str(roundIndex))
         if roundIndex > 0:
            roundName = self.roundNames[roundIndex-1]
#           print("::: GCP newRoundName "+roundName)
            fitPower = teamObject.GetFitPower(year,roundName)
         else:
                fitPower = {}
                fitPower["offense"] = teamObject.GetOffensePower(year-1)
                fitPower["defense"] = teamObject.GetDefensePower(year-1) 
      return fitPower

   def FitPowerForCurrentRound(self, learningRate=0):
      print("Fitting power for season "+str(self.seasonYear)+ ", round "+self.roundNames[self.currentRound])
      year = self.seasonYear
        
      # Note that it is assumed that the current power for each team is synced to currentRound
        
      # Structure to store fit power for each game.
      # powerPerTeamToDate[teamId][gameId][{offenseAdjusted, defenseAdjusted, gameType (conference, nonconfence, nondivision)}]
      powerPerTeamToDate = {}

      # For each round from beginning to this round, for each game, compute fit power (starting from current power).
      for iRound in range(self.currentRound+1):
#            print("::: fitPower for round "+self.roundNames[iRound])
         if (self.seasonInfo.get(self.roundNames[iRound]) != None):
            for game in self.seasonInfo[self.roundNames[iRound]]:
#              game.PrintGame() # :::
               # Perform fit of power to match actual score and store results in PowerPerTeamToDate
               gameType = "TBD"
               if (game.team1Object.GetConferenceName(year) == game.team2Object.GetConferenceName(year)):
                  gameType = "conference"
               elif (game.team1Object.GetDivisionName(year) == game.team2Object.GetDivisionName(year)):
                  gameType = "nonconference"
               else:
                  gameType = "nondivision"
               # For computing new fit to power, start with power from previous year for each call to FitPowerForCurrentRound.
               team1Power = {}
               team1Power["offense"] = game.team1Object.GetOffensePower(year-1)
               team1Power["defense"] = game.team1Object.GetDefensePower(year-1)
               team2Power = {}
               team2Power["offense"] = game.team2Object.GetOffensePower(year-1)
               team2Power["defense"] = game.team2Object.GetDefensePower(year-1)

               # Note that predictions for the coming week should use the latest fit power (i.e. the best currently derived 
               # power for the season to-date)
               #team1Power = self.GetCurrentPower(game.team1Object, year, self.roundNames[currentRound])
               #team2Power = self.GetCurrentPower(game.team2Object, year, self.roundNames[currentRound])

#              expectedResult = self.gameSimulator.GetScoreAndProbability(team1Power["offense"], team1Power["defense"], 
#                                                                     team2Power["offense"], team2Power["defense"], 
#                                                                     game.homeField) # :::
#              print("::: expected "+str(expectedResult))
               adjustedPower = self.gameSimulator.AdjustPowerToFitActualScore(team1Power["offense"], team1Power["defense"], 
                                                                     team2Power["offense"], team2Power["defense"], 
                                                                     game.homeField, game.score1, game.score2)
               if (powerPerTeamToDate.get(game.team1Object.teamName) == None):
                  powerPerTeamToDate[game.team1Object.teamName] = {}
               powerPerTeamToDate[game.team1Object.teamName][game.gameId] = {}
               powerPerTeamToDate[game.team1Object.teamName][game.gameId]["offenseAdjusted"] = adjustedPower["offense1"]
               powerPerTeamToDate[game.team1Object.teamName][game.gameId]["defenseAdjusted"] = adjustedPower["defense1"]
               powerPerTeamToDate[game.team1Object.teamName][game.gameId]["offenseBase"] = team1Power["offense"]
               powerPerTeamToDate[game.team1Object.teamName][game.gameId]["defenseBase"] = team1Power["defense"]
               powerPerTeamToDate[game.team1Object.teamName][game.gameId]["gameType"] = gameType
               if (powerPerTeamToDate.get(game.team2Object.teamName) == None):
                  powerPerTeamToDate[game.team2Object.teamName] = {}
               powerPerTeamToDate[game.team2Object.teamName][game.gameId] = {}
               powerPerTeamToDate[game.team2Object.teamName][game.gameId]["offenseAdjusted"] = adjustedPower["offense2"]
               powerPerTeamToDate[game.team2Object.teamName][game.gameId]["defenseAdjusted"] = adjustedPower["defense2"]
               powerPerTeamToDate[game.team2Object.teamName][game.gameId]["offenseBase"] = team2Power["offense"]
               powerPerTeamToDate[game.team2Object.teamName][game.gameId]["defenseBase"] = team2Power["defense"]
               powerPerTeamToDate[game.team2Object.teamName][game.gameId]["gameType"] = gameType
#              print ("::: t1 "+str(powerPerTeamToDate[game.team1Object.teamName][game.gameId])+" t2 "
#                           +str(powerPerTeamToDate[game.team2Object.teamName][game.gameId]))
                  
      meanSquarePowerError = {}
      meanSquarePowerError["offense"] = 0
      meanSquarePowerError["defense"]= 0
      meanSquarePowerError["count"] = 0                
        
      # For each team in PowerPerTeamToDate
      for team in powerPerTeamToDate.keys():
         # Average the power over all games for the team
         offenseTotal = 0
         defenseTotal = 0
         weightTotal = 0
         baseWeight = -1
         numberOfGames = len(powerPerTeamToDate[team])
         weightIndex = int((numberOfGames-1)/2)
         for gameId in powerPerTeamToDate[team].keys():
            if (baseWeight == -1):
               baseWeight = self.gameSimulator.weights["previousSeason"][weightIndex]
               weightTotal += baseWeight
               offenseTotal += powerPerTeamToDate[team][gameId]["offenseBase"]*baseWeight
               defenseTotal += powerPerTeamToDate[team][gameId]["defenseBase"]*baseWeight
            gameWeight = self.gameSimulator.weights[powerPerTeamToDate[team][gameId]["gameType"]][weightIndex]
            offenseTotal += powerPerTeamToDate[team][gameId]["offenseAdjusted"]*gameWeight
            defenseTotal += powerPerTeamToDate[team][gameId]["defenseAdjusted"]*gameWeight
            weightTotal += gameWeight
#           print("::: gameWeight "+str(gameWeight)+" gameType "+powerPerTeamToDate[team][gameId]["gameType"])
         offenseUpdated = offenseTotal/weightTotal
         defenseUpdated = defenseTotal/weightTotal
#        print("::: team "+team+" o&d "+str(offenseUpdated)+" "+str(defenseUpdated))
            
         # Update team power
         team.Team.teamObjectByName[team].SetFitPower(year, self.roundNames[self.currentRound], offenseUpdated, defenseUpdated)
         # If actual power given, update meanSquarePowerError
         actualPower = team.Team.teamObjectByName[team].GetPowerActual(year)
         if (actualPower["offense"] != None):
            meanSquarePowerError["offense"] += (offenseUpdated-actualPower["offense"])**2
            meanSquarePowerError["defense"] += (defenseUpdated-actualPower["defense"])**2
         meanSquarePowerError["count"] += 1
            
         if (learningRate > 0):
            # Update the deltas for each weight.
            #
            #   Cost function is the squared error of (actual-derived power)^2
            #
            #   cost for a given week = sum over each team_i of [actualOffense_i - (sum over each game_ij of weight_ij*adjustedOffense_ij)/totalWeight_i]^2
            #                                                   [actualDefense_i - (sum over each game_ij of weight_ij*adjustedDefense_ij)/totalWeight_i]^2
            #
            #   Update the weights as follows (e.g. like http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks Backpropagation Algorithm)
            #
            #      weight_lk_new = weight_lk - learningRate * d cost/d weight_lk
            #
            #      d cost/d weight_lk  = sum over each team_i of 
            #         2*[actualOffense_i - (sum over each game_ij of weight_ij*adjustedOffense_ij)/totalWeight_i]*[sum over each game_ij of -adjustedOffense_ij*d weight_ij/d weight_lk]
            #         + 2*[actualDefense_i - (sum over each game_ij of weight_ij*adjustedDefense_ij)/totalWeight_i]*[sum over each game_ij of -adjustedDefense_ij*d weight_ij/d weight_lk]
            #
            #         where d weight_ij/d weight_lk is 0 when ij != lk and d weight_lk/d weigh_lk = 1
            #
            #         note that assuming that totalWeight is approximately constant, i.e. each weight's change is small with respect to the total weight
            #
            #      d cost/d weight_lk  = sum over each team_i of 
            #         2*[actualOffense_i - (sum over each game_ij of weight_ij*adjustedOffense_ij)/totalWeight_i]*[sum over each game with weight_lk_of -adjustedOffense_ij/totalWeight_i]
            #         + 2*[actualDefense_i - (sum over each game_ij of weight_ij*adjustedDefense_ij)/totalWeight_i]*[sum over each game with weight_lk_of -adjustedDefense_ij/totalWeight_i]
            #
            #   To do the update, for each week w, add the following to the deltaWeight_lk:
            #
            #      For each team i:
            #         deltaWeight_lk -= [actualOffense_iw - (sum over each game_ij of weight_ij*adjustedOffense_ij)/totalWeight_iw]*[sum over each game_ij using weight_lk of adjustedOffense_ijw/totalWeight_i]
            #                         + [actualDefense_iw - (sum over each game_ij of weight_ij*adjustedDefense_ij)/totalWeight_iw]*[sum over each game_ij using weight_lk of adjustedDefense_ijw/totalWeight_i]
            baseCount = 0
            for gameId in powerPerTeamToDate[team].keys():
               self.gameSimulator.deltaWeights[powerPerTeamToDate[team][gameId]["gameType"]][weightIndex] -= learningRate/weightTotal*(
                        (actualPower["offense"]-offenseUpdated)*powerPerTeamToDate[team][gameId]["offenseAdjusted"]
                        + (actualPower["defense"]-defenseUpdated)*powerPerTeamToDate[team][gameId]["defenseAdjusted"])
               if (baseCount == 0):
                  baseCount = 1
                  self.gameSimulator.deltaWeights["previousSeason"][weightIndex] -= learningRate/weightTotal*(
                           (actualPower["offense"]-offenseUpdated)*powerPerTeamToDate[team][gameId]["offenseBase"]
                           + (actualPower["defense"]-defenseUpdated)*powerPerTeamToDate[team][gameId]["defenseBase"])
      
      # Compute and print RMS of power
      if (meanSquarePowerError["count"] > 0):
         # Print RMS
         offenseRms = math.sqrt(meanSquarePowerError["offense"]/meanSquarePowerError["count"])
         defenseRms = math.sqrt(meanSquarePowerError["defense"]/meanSquarePowerError["count"])
         print("::: RMS errors for round "+self.roundNames[self.currentRound]+" "+str(offenseRms)+" "+str(defenseRms))
      
   def RandomizeTeamPower(self, maxOffense, minOffense, maxDefense, minDefense):
      activeTeams = team.Team.GetListOfTeamIds(self.seasonYear)
      for teamId in activeTeams:
         teamObject = team.Team.teamObjectById[teamId]
         teamObject.SetPower(self.seasonYear, minOffense+(maxOffense-minOffense)*random.random(), minDefense+(maxDefense-minDefense)*random.random())
      #for iRound in range(len(self.roundNames)):
      #   if (self.seasonInfo.get(self.roundNames[iRound]) != None):
      #      for game in self.seasonInfo[self.roundNames[iRound]]:
      #         game.team1Object.SetPower(self.seasonYear, minOffense+(maxOffense-minOffense)*random.random(), minDefense+(maxDefense-minDefense)*random.random())

   def AdjustPowerAndLinearParametersFromScores(self, learningRate, adamUpdateEnable, parameters=None, averageTeamPower=15.37, powerCostFactor=0.001, verbose=False, constrainAveragePower=True, forceAverageOffense=None, forceAverageDefense=None, roundStopCount=None):
      # Before running this for the first time, it is recommended to RandomizeTeamPower and gameSimulator.RandomizeLinearParameters
        
      # Cost as (s1-s2-actualScore1+actualScore2)^2 + deprioritizeTotalForCost*(total_no_spread-min(actualScore1+homeAdjust,actualScore2-homeAdjust))^2:
      #      Note that homeFieldAdvantage is assumed to not apply to the total point, rather to the spread only.
      #      It does appear in total component though to convert actual score to neutral field scores.
      #      homeAdjust = team1 home: 0.5*homeFieldAdvantage
      #                   team2 home: -0.5*homeFieldAdvantage
      #                   else: 0
      #
      #   spread = s1 - s2 = 0.5*(totCoefOff*(o1+o2) + totCoefDef*(d1+d2) + totCoef0 + spdCoefOff*(o1-o2) + spdCoefDef*(d1-d2) + 0.5*homeFieldAdvantage)
      #                    - 0.5*(totCoefOff*(o2+o1) + totCoefDef*(d2+d1) + totCoef0 + spdCoefOff*(o2-o1) + spdCoefDef*(d2-d1) - 0.5*homeFieldAdvantage)
      #          = spdCoefOff*(o1-o2) + spdCoefDef(d1-d2) + homeFieldAdvantage
      #   total  = s1 + s2 = totCoefOff*(o1+o2) + totCoefDef*(d1+d2) + totCoef0 + abs(spread)
      #   total_no_spread = totCoefOff*(o1+o2) + totCoefDef*(d1+d2) + totCoef0
      #
      #   spread = spdCoefOff*(o1-o2) + spdCoefDef*(d1-d2) + homeFieldAdvantage
      #   spread_neurtral = spdCoefOff*(o1-o2) + spdCoefDef*(d1-d2)
      #      note that homeFieldAdvantage is assumed to not apply to the total point, rather to the spread only.
      #   total  = totCoefOff*(o1+o2) + totCoefDef*(d1+d2) + totCoef0 + abs(spread_neutral)
      #          = spreadSign * (spdCoefOff*(o1-o2) + spdCoefDef*(d1-d2)) + totCoefOff*(o1+o2) + totCoefDef*(d1+d2) + totCoef0
      #            spreadSign = 1 if spread_neutral >= 0, else -1
      #   total_no_spread = totCoefOff*(o1+o2) + totCoefDef*(d1+d2) + totCoef0
      #
      #  For spread, let offDiff = o1 - o2.  d o1 = -d o2, defDiff = d1 - d2, d d1 = -d d2
      #  For total, let offTotal = o1 + o2.  d o1 = d o2, defTotal = d1 + d2, d d1 = d d2
      #
      # cost function = sum over all games_ij of (s1_ij - s2_ij - actualScore1_ij + actualScore2_ij)^2 + deprioritizeTotalForCost*(s1_ij + s2_ij - min(actualScore1_ij+homeAdjust, actualScore2_ij-homeAdjust))^2
      #               = sum over all games_ij of (spdCoefOff*(off_i-off_j) + spdCoefDef*(def_i-def_j) + homeFieldAdvantage - actualScore1_ij + actualScore2_ij)^2
      #               + deprioritizeTotalForCost*(totCoefOff*(off_i+off_j) + totCoefDef*(def_i+def_j) + totCoef0 - min(actualScore1_ij+homeAdjust,actualScore2_ij-homeAdjust))^2
      #
      #               = sum over all games_ij of spreadDiff_ij^2 + deprioritizeTotalForCost*totalNoSpreadDiff_ij^2
      #
      #    spreadDiff_ij = s1_ij - s2_ij - actualScore1_ij + actualScore2_ij
      #    totalNoSpreadDiff_ij = s1_ij + s2_ij - min(actualScore1_ij+homeAdjust, actualScore2_ij-homeAdjust)

      #   For team1:
      #      d cost/d off_i = sum over games involving off_i of 2*(spdCoefOff*(off_i-off_j) + spdCoefDef*(def_i-def_j) + homeFieldAdvantage - actualScore1_ij + actualScore2_ij)*spdCoedOff + deprioritizeTotalForCost*2*(totCoefOff*(off_i+off_j) + totCoefDef*(def_i+def_j) + totCoef0 - min(actualScore1_ij, actualScore2_ij)) * totCoefOff
      #                     = sum over games involving off_i of 2*spreadDiff_ij*spdCoedOff + deprioritizeTotalForCost*2*totalNoSpreadDiff_ij*totCoefOff
      #
      #      d cost/d def_i = sum over games involving def_i of 2*(spdCoefOff*(off_i-off_j) + spdCoefDef*(def_i-def_j) + homeFieldAdvantage - actualScore1_ij + actualScore2_ij)*spdCoefDef + deprioritizeTotalForCost*2*(totCoefOff*(off_i+off_j) + totCoefDef*(def_i+def_j) + totCoef0 - min(actualScore1_ij, actualScore2_ij)) * totCoefDef)
      #                     = sum over games involving def_i of 2*spreadDiff_ij*spdCoedDef + deprioritizeTotalForCost*2*totalNoSpreadDiff_ij*totCoefDef
      #
      #   For team2:
      #      d cost/d off_j = sum over games involving off_j of 2*(spdCoefOff*(off_i-off_j) + spdCoefDef*(def_i-def_j) + homeFieldAdvantage - actualScore1_ij + actualScore2_ij)*-spdCoedOff + deprioritizeTotalForCost*2*(totCoefOff*(off_i+off_j) + totCoefDef*(def_i+def_j) + totCoef0 - min(actualScore1_ij, actualScore2_ij)) * totCoefOff
      #                     = sum over games involving off_i of 2*spreadDiff_ij*-spdCoedOff + deprioritizeTotalForCost*2*totalNoSpreadDiff_ij*totCoefOff
      #
      #      d cost/d def_j = sum over games involving def_i of 2*(spdCoefOff*(off_i-off_j) + spdCoefDef*(def_i-def_j) + homeFieldAdvantage - actualScore1_ij + actualScore2_ij)*-spdCoefDef + deprioritizeTotalForCost*2*(totCoefOff*(off_i+off_j) + totCoefDef*(def_i+def_j) + totCoef0 - min(actualScore1_ij, actualScore2_ij)) * totCoefDef
      #                     = sum over games involving def_i of 2*spreadDiff_ij*-spdCoedDef + deprioritizeTotalForCost*2*totalNoSpreadDiff_ij*totCoefDef
      #   
      #      d cost/d spdCoefOff = sum over games of 2*spreadDiff_ij*(off_i-off_j)
      #
      #      d cost/d spdCoefDef = sum over games of 2*spreadDiff_ij*(def_i-def_j)
      #
      #      d cost/d totCoefOff = sum over games of 2*deprioritizeTotalForCost*totalNoSpreadDiff_ij*(off_i+off_j)
      #
      #      d cost/d totCoefDef = sum over games of 2*deprioritizeTotalForCost*totalNoSpreadDiff_ij*(def_i+def_j)
      #
      #      d cost/d totCoef0 = sum over games of 2*deprioritizeTotalForCost*totalNoSpreadDiff_ij
      #
      #   if actualScore1_ij+homeAdjust <= actualScore2-homeAdjust
      #      d cost/d homeFieldAdvantage = sum over games_ij of 
      #         2*spreadDiff_ij*2*d homeAdjust/d homeFieldAdvantage + 2*deprioritizeTotalForCost*totalNoSpreadDiff_ij*d homeAdjust/d homeFieldAdvantage
      #   else
      #      d cost/d homeFieldAdvantage = sum over games_ij of 
      #         2*spreadDiff_ij*2*d homeAdjust/d homeFieldAdvantage + 2*deprioritizeTotalForCost*totalNoSpreadDiff_ij*d -homeAdjust/d homeFieldAdvantage
      #   

      # Fill in default values for parameters
      defaultParameters = {'teamPowers': True, 'homeFieldAdvantage': True, 'deprioritizeTotalForCost': 0.25, 'linearParameters':True}
      # if deprioritizeTotalForCost == 0 then do the cost with respect to (s1-actualScore1)^2 + (s2-actualScore2)^2.
      # if deprioritiezTotalForCost > 0 then do the cost with respect to (spread-actualSpread)^2 + deprioritizeTotalForCost*(total-actualTotal)^2
      if (parameters == None):
         parameters = defaultParameters
      else:
         for param in defaultParameters.keys():
            if (parameters.get(param) == None):
               parameters[param] = defaultParameters[param]
        
      if (verbose):
         print("AdjustPowerAndLinearParametersFromScores: parameters "+str(parameters))
        
      teamPowerDeltas = {}
      # teamPowerDeltas[teamName][offense]
      # teamPowerDeltas[teamName][defense]
      linearDeltas = {}
      linearDeltas["spdCoefOff"] = 0
      linearDeltas["spdCoefDef"] = 0
      linearDeltas["totCoefOff"] = 0
      linearDeltas["totCoefDef"] = 0
      linearDeltas["totCoef0"] = 0
      linearDeltas["homeFieldAdvantage"] = 0
      linearDeltas["count"]= 0
        
      costOrig = 0
      countOrig = 0
      if (roundStopCount == None):
         roundStopCount = len(self.roundNames)
      #for iRound in range(len(self.roundNames)):
      for iRound in range(roundStopCount):
         if (self.seasonInfo.get(self.roundNames[iRound]) != None):
            for game in self.seasonInfo[self.roundNames[iRound]]:
               if (teamPowerDeltas.get(game.team1Object.teamName) == None):
                  teamPowerDeltas[game.team1Object.teamName] = {}
                  teamPowerDeltas[game.team1Object.teamName]["offense"] = 0
                  teamPowerDeltas[game.team1Object.teamName]["defense"] = 0
                  teamPowerDeltas[game.team1Object.teamName]["count"] = 0
                  teamPowerDeltas[game.team1Object.teamName]["offenseDs"] = 0
                  teamPowerDeltas[game.team1Object.teamName]["offenseDt"] = 0
               if (teamPowerDeltas.get(game.team2Object.teamName) == None):
                  teamPowerDeltas[game.team2Object.teamName] = {}
                  teamPowerDeltas[game.team2Object.teamName]["offense"] = 0
                  teamPowerDeltas[game.team2Object.teamName]["defense"] = 0
                  teamPowerDeltas[game.team2Object.teamName]["count"] = 0
                  teamPowerDeltas[game.team2Object.teamName]["offenseDs"] = 0
                  teamPowerDeltas[game.team2Object.teamName]["offenseDt"] = 0
               power1 = game.team1Object.GetPower(self.seasonYear)
               #power1a = game.team1Object.GetPower(self.seasonYear-1)
               power2 = game.team2Object.GetPower(self.seasonYear)
               #power2a = game.team2Object.GetPower(self.seasonYear-1)
               #print("::: year "+str(self.seasonYear)+" team1 "+game.team1Object.teamName+" team2 "+game.team2Object.teamName)
               #print(":::-1 team1Name "+game.team1Object.teamName+" prevYr "+str(power1a)+" cur "+str(power1))
               #print(":::-1 team2Name "+game.team2Object.teamName+" prevYr "+str(power2a)+" cur "+str(power2))
               scores = self.gameSimulator.GetScores(
                        power1["offense"], power1["defense"], power2["offense"], power2["defense"], game.homeField, (countOrig == 0))
               totalNoSpreadResults = self.gameSimulator.GetTotalNoSpread(power1["offense"], power1["defense"], power2["offense"], power2["defense"], game.score1, game.score2, game.homeField, (countOrig == 0))
               actualTotalNoSpread = totalNoSpreadResults["actualTotalNoSpread"]
               totalNoSpread = totalNoSpreadResults["totalNoSpread"]
               teamPowerDeltas[game.team1Object.teamName]["offenseDs"] += 2*(
                        self.gameSimulator.spdCoefOff * (scores["score1"] - scores["score2"] - game.score1 + game.score2))
               teamPowerDeltas[game.team1Object.teamName]["offenseDt"] += 2*(
                        + self.gameSimulator.totCoefOff * (totalNoSpread - actualTotalNoSpread) * parameters["deprioritizeTotalForCost"])
               teamPowerDeltas[game.team1Object.teamName]["offense"] += 2*(
                        self.gameSimulator.spdCoefOff * (scores["score1"] - scores["score2"] - game.score1 + game.score2)
                        + self.gameSimulator.totCoefOff * (totalNoSpread - actualTotalNoSpread) * parameters["deprioritizeTotalForCost"])
               teamPowerDeltas[game.team1Object.teamName]["defense"] += 2*( 
                        self.gameSimulator.spdCoefDef * (scores["score1"] - scores["score2"] - game.score1 + game.score2)
                        + self.gameSimulator.totCoefDef * (totalNoSpread - actualTotalNoSpread) * parameters["deprioritizeTotalForCost"])
               teamPowerDeltas[game.team2Object.teamName]["offense"] += 2*(
                        -self.gameSimulator.spdCoefOff * (scores["score1"] - scores["score2"] - game.score1 + game.score2)
                        + self.gameSimulator.totCoefOff * (totalNoSpread - actualTotalNoSpread) * parameters["deprioritizeTotalForCost"])
               teamPowerDeltas[game.team2Object.teamName]["defense"] += 2*( 
                        -self.gameSimulator.spdCoefDef * (scores["score1"] - scores["score2"] - game.score1 + game.score2)
                        + self.gameSimulator.totCoefDef * (totalNoSpread - actualTotalNoSpread) * parameters["deprioritizeTotalForCost"])
               teamPowerDeltas[game.team1Object.teamName]["count"] += 1
               teamPowerDeltas[game.team2Object.teamName]["count"] += 1
               linearDeltas["spdCoefOff"] += 2*( 
                        (power1["offense"]-power2["offense"])
                        * (scores["score1"] - scores["score2"] - game.score1 + game.score2)
                        )
               linearDeltas["spdCoefDef"] += 2*( 
                        (power1["defense"]-power2["defense"])
                        * (scores["score1"] - scores["score2"] - game.score1 + game.score2)
                        )
               linearDeltas["totCoefOff"] += 2*( 
                        (power1["offense"]+power2["offense"])
                        * (totalNoSpread - actualTotalNoSpread) * parameters["deprioritizeTotalForCost"]
                        )
               linearDeltas["totCoefDef"] += 2*( 
                        (power1["defense"]+power2["defense"])
                        * (totalNoSpread - actualTotalNoSpread) * parameters["deprioritizeTotalForCost"]
                        )
               linearDeltas["totCoef0"] += 2*( 
                        (totalNoSpread - actualTotalNoSpread) * parameters["deprioritizeTotalForCost"]
                        )
               #TODO - update: linearDeltas["homeFieldAdvantage"] += (scores["score1"] - scores["score2"] - game.score1 + game.score2)
               #   if actualScore1_ij+homeAdjust <= actualScore2-homeAdjust
               #      d cost/d homeFieldAdvantage = sum over games_ij of 
               #         2*spreadDiff_ij*2*d homeAdjust/d homeFieldAdvantage + 2*deprioritizeTotalForCost*totalNoSpreadDiff_ij*d homeAdjust/d homeFieldAdvantage
               #   else
               #      d cost/d homeFieldAdvantage = sum over games_ij of 
               #         2*spreadDiff_ij*2*d homeAdjust/d homeFieldAdvantage + 2*deprioritizeTotalForCost*totalNoSpreadDiff_ij*d -homeAdjust/d homeFieldAdvantage
               linearDeltas["count"] += 1

               #if (costOrig == 0):
               #   actualPower1 = game.team1Object.GetPowerActual(self.seasonYear)
               #   actualPower2 = game.team2Object.GetPowerActual(self.seasonYear)
               #   # print out first game as a sample
               #   print("::: costOrig "+str(scores)+" actual "+str(game.score1)+" "+str(game.score2)+" "+str(power1)+" "+str(power2)+" act-cur"+" dd1 "+str(actualPower1["defense"]-power1["defense"])+" do1 "+str(actualPower1["offense"]-power1["offense"])+" dd2 "+str(actualPower2["defense"]-power2["defense"])+" do2 "+str(actualPower2["offense"]-power2["offense"]))
               countOrig += 1
               if (parameters["deprioritizeTotalForCost"] == 0):
                  costOrig += (scores["score1"] - game.score1)**2 + (scores["score2"] - game.score2)**2
               else:
                  costOrig += ((scores["score1"] - scores["score2"] - game.score1 + game.score2)**2
                        + parameters["deprioritizeTotalForCost"]*(scores["score1"] + scores["score2"] - game.score1 - game.score2)**2)
               #print("::: costOrig in progress "+str(costOrig))
        
      if (countOrig > 0):
         if (adamUpdateEnable and (parameters["linearParameters"] or parameters["homeFieldAdvantage"])):
            # Add one power of t since doing one time step
            self.gameSimulator.adamParams["beta1t"] *= self.gameSimulator.adamParams["beta1"]
            self.gameSimulator.adamParams["beta2t"] *= self.gameSimulator.adamParams["beta2"]

         if (parameters["linearParameters"]):
            deltaSpdCoefOff = linearDeltas["spdCoefOff"]/linearDeltas["count"]
            deltaSpdCoefDef = linearDeltas["spdCoefDef"]/linearDeltas["count"]
            deltaTotCoefOff = linearDeltas["totCoefOff"]/linearDeltas["count"]
            deltaTotCoefDef = linearDeltas["totCoefDef"]/linearDeltas["count"]
            deltaTotCoef0 = linearDeltas["totCoef0"]/linearDeltas["count"]
            #print("::: deltaTotCoefOff="+str(deltaTotCoefOff)+" deltaTotCoefDef="+str(deltaTotCoefDef))
            if (adamUpdateEnable):
               self.gameSimulator.adamMoments["spdCoefOff"]["m"] = (self.gameSimulator.adamParams["beta1"]*self.gameSimulator.adamMoments["spdCoefOff"]["m"]
                  + (1.0 - self.gameSimulator.adamParams["beta1"])*deltaSpdCoefOff)
               self.gameSimulator.adamMoments["spdCoefOff"]["v"] = (self.gameSimulator.adamParams["beta2"]*self.gameSimulator.adamMoments["spdCoefOff"]["v"]
                  + (1.0 - self.gameSimulator.adamParams["beta2"])*deltaSpdCoefOff*deltaSpdCoefOff)
               deltaSpdCoefOff = ((self.gameSimulator.adamMoments["spdCoefOff"]["m"] / (1.-self.gameSimulator.adamParams["beta1t"]))
                  / (math.sqrt(self.gameSimulator.adamMoments["spdCoefOff"]["v"] / (1.-self.gameSimulator.adamParams["beta2t"])) 
                  + self.gameSimulator.adamParams["epsilon"]))
               self.gameSimulator.adamMoments["spdCoefDef"]["m"] = (self.gameSimulator.adamParams["beta1"]*self.gameSimulator.adamMoments["spdCoefDef"]["m"]
                  + (1.0 - self.gameSimulator.adamParams["beta1"])*deltaSpdCoefDef)
               self.gameSimulator.adamMoments["spdCoefDef"]["v"] = (self.gameSimulator.adamParams["beta2"]*self.gameSimulator.adamMoments["spdCoefDef"]["v"]
                  + (1.0 - self.gameSimulator.adamParams["beta2"])*deltaSpdCoefDef*deltaSpdCoefDef)
               deltaSpdCoefDef = ((self.gameSimulator.adamMoments["spdCoefDef"]["m"] / (1.-self.gameSimulator.adamParams["beta1t"]))
                  / (math.sqrt(self.gameSimulator.adamMoments["spdCoefDef"]["v"] / (1.-self.gameSimulator.adamParams["beta2t"])) 
                  + self.gameSimulator.adamParams["epsilon"]))
               self.gameSimulator.adamMoments["totCoefOff"]["m"] = (self.gameSimulator.adamParams["beta1"]*self.gameSimulator.adamMoments["totCoefOff"]["m"]
                  + (1.0 - self.gameSimulator.adamParams["beta1"])*deltaTotCoefOff)
               self.gameSimulator.adamMoments["totCoefOff"]["v"] = (self.gameSimulator.adamParams["beta2"]*self.gameSimulator.adamMoments["totCoefOff"]["v"]
                  + (1.0 - self.gameSimulator.adamParams["beta2"])*deltaTotCoefOff*deltaTotCoefOff)
               deltaTotCoefOff = ((self.gameSimulator.adamMoments["totCoefOff"]["m"] / (1.-self.gameSimulator.adamParams["beta1t"]))
                  / (math.sqrt(self.gameSimulator.adamMoments["totCoefOff"]["v"] / (1.-self.gameSimulator.adamParams["beta2t"])) 
                  + self.gameSimulator.adamParams["epsilon"]))
               self.gameSimulator.adamMoments["totCoefDef"]["m"] = (self.gameSimulator.adamParams["beta1"]*self.gameSimulator.adamMoments["totCoefDef"]["m"]
                  + (1.0 - self.gameSimulator.adamParams["beta1"])*deltaTotCoefDef)
               self.gameSimulator.adamMoments["totCoefDef"]["v"] = (self.gameSimulator.adamParams["beta2"]*self.gameSimulator.adamMoments["totCoefDef"]["v"]
                  + (1.0 - self.gameSimulator.adamParams["beta2"])*deltaTotCoefDef*deltaTotCoefDef)
               deltaTotCoefDef = ((self.gameSimulator.adamMoments["totCoefDef"]["m"] / (1.-self.gameSimulator.adamParams["beta1t"]))
                  / (math.sqrt(self.gameSimulator.adamMoments["totCoefDef"]["v"] / (1.-self.gameSimulator.adamParams["beta2t"])) 
                  + self.gameSimulator.adamParams["epsilon"]))
               self.gameSimulator.adamMoments["totCoef0"]["m"] = (self.gameSimulator.adamParams["beta1"]*self.gameSimulator.adamMoments["totCoef0"]["m"]
                  + (1.0 - self.gameSimulator.adamParams["beta1"])*deltaTotCoef0)
               self.gameSimulator.adamMoments["totCoef0"]["v"] = (self.gameSimulator.adamParams["beta2"]*self.gameSimulator.adamMoments["totCoef0"]["v"]
                  + (1.0 - self.gameSimulator.adamParams["beta2"])*deltaTotCoef0*deltaTotCoef0)
               deltaTotCoef0 = ((self.gameSimulator.adamMoments["totCoef0"]["m"] / (1.-self.gameSimulator.adamParams["beta1t"]))
                  / (math.sqrt(self.gameSimulator.adamMoments["totCoef0"]["v"] / (1.-self.gameSimulator.adamParams["beta2t"])) 
                  + self.gameSimulator.adamParams["epsilon"]))
            self.gameSimulator.spdCoefOff -= learningRate * deltaSpdCoefOff
            self.gameSimulator.spdCoefDef -= learningRate * deltaSpdCoefDef
            self.gameSimulator.totCoefOff -= learningRate * deltaTotCoefOff
            self.gameSimulator.totCoefDef -= learningRate * deltaTotCoefDef
            self.gameSimulator.totCoef0 -= learningRate * deltaTotCoef0
            if (self.gameSimulator.spdCoefOff < 0.001):
               self.gameSimulator.spdCoefOff = 0.001
            if (self.gameSimulator.spdCoefDef < 0.001):
               self.gameSimulator.spdCoefDef = 0.001
            if (self.gameSimulator.totCoefOff < 0.001):
               self.gameSimulator.totCoefOff = 0.001
            if (self.gameSimulator.totCoefDef > -0.001):
               self.gameSimulator.totCoefDef = -0.001

         if (parameters["homeFieldAdvantage"]):
            print("fit for homeFieldAdvantage currently needs and update")
            exit(1)
            deltaHomeFieldAdvantage = linearDeltas["homeFieldAdvantage"]/linearDeltas["count"]
            if (adamUpdateEnable):
               self.gameSimulator.adamMoments["homeFieldAdvantage"]["m"] = (self.gameSimulator.adamParams["beta1"]*self.gameSimulator.adamMoments["homeFieldAdvantage"]["m"]
                  + (1.0 - self.gameSimulator.adamParams["beta1"])*deltaHomeFieldAdvantage)
               self.gameSimulator.adamMoments["homeFieldAdvantage"]["v"] = (self.gameSimulator.adamParams["beta2"]*self.gameSimulator.adamMoments["homeFieldAdvantage"]["v"]
                  + (1.0 - self.gameSimulator.adamParams["beta2"])*deltaHomeFieldAdvantage*deltaHomeFieldAdvantage)
               deltaHomeFieldAdvantage = ((self.gameSimulator.adamMoments["homeFieldAdvantage"]["m"] / (1.-self.gameSimulator.adamParams["beta1t"]))
                  / (math.sqrt(self.gameSimulator.adamMoments["homeFieldAdvantage"]["v"] / (1.-self.gameSimulator.adamParams["beta2t"])) 
                  + self.gameSimulator.adamParams["epsilon"]))
            self.gameSimulator.homeFieldAdvantage -= learningRate * deltaHomeFieldAdvantage

         if (parameters["teamPowers"]):
            offensePowerDiff = 0
            defensePowerDiff = 0
            currentAveTeamPower = 0
            teamCount = 0
            currentAverageOffense = 0
            currentAverageDefense = 0
            teamCount = 0
            for teamName in teamPowerDeltas.keys():
               teamObject = team.Team.teamObjectByName[teamName]
               power = teamObject.GetPower(self.seasonYear)
               teamCount += 1
               currentAveTeamPower += power["offense"] + (-self.gameSimulator.totCoefOff/self.gameSimulator.totCoefDef)*power["defense"]
               currentAverageOffense += power["offense"]
               currentAverageDefense += power["defense"]
            currentAveTeamPower /= teamCount
            currentAverageOffense /= teamCount
            currentAverageDefense /= teamCount
            if (constrainAveragePower):
               deltaAvePowerOff = (currentAveTeamPower - averageTeamPower)/(1-(self.gameSimulator.totCoefOff/self.gameSimulator.totCoefDef))
               deltaAvePowerDef = (currentAveTeamPower - averageTeamPower)*(self.gameSimulator.totCoefOff/self.gameSimulator.totCoefDef)/(1-(self.gameSimulator.totCoefOff/self.gameSimulator.totCoefDef))
               print("::: Current Team Power "+str(currentAveTeamPower)+" Target Team Power "+str(averageTeamPower)+" deltaOff "+str(deltaAvePowerOff)+" deltaDef "+str(deltaAvePowerDef))
            else:
               deltaAvePowerOff = 0.0
               deltaAvePowerDef = 0.0
               if (forceAverageOffense != None):
                  deltaAvePowerOff = (currentAverageOffense - forceAverageOffense)
                  print("::: currentAverageOffense="+str(currentAverageOffense)+" forceDelta "+str(deltaAvePowerOff))
               if (forceAverageDefense != None):
                  deltaAvePowerDef = (currentAverageDefense - forceAverageDefense)
                  print("::: currentAverageDefense="+str(currentAverageDefense)+" forceDelta "+str(deltaAvePowerDef))
            for teamName in teamPowerDeltas.keys():
               teamObject = team.Team.teamObjectByName[teamName]
               power = teamObject.GetPower(self.seasonYear)
               deltaOffense = teamPowerDeltas[teamObject.teamName]["offense"]/teamPowerDeltas[teamObject.teamName]["count"]
               deltaDefense = teamPowerDeltas[teamObject.teamName]["defense"]/teamPowerDeltas[teamObject.teamName]["count"]
               if (adamUpdateEnable):
                  #deltaOffense0 = deltaOffense
                  #deltaDefense0 = deltaDefense
                  powerMoments = teamObject.GetAdamMoments(self.seasonYear)
                  powerMoments["beta1t"] *= self.gameSimulator.adamParams["beta1"]
                  powerMoments["beta2t"] *= self.gameSimulator.adamParams["beta2"]
                  powerMoments["offenseMoment1"] = (self.gameSimulator.adamParams["beta1"]*powerMoments["offenseMoment1"]
                     + (1.0 - self.gameSimulator.adamParams["beta1"])*deltaOffense)
                  powerMoments["offenseMoment2"] = (self.gameSimulator.adamParams["beta2"]*powerMoments["offenseMoment2"]
                     + (1.0 - self.gameSimulator.adamParams["beta2"])*deltaOffense*deltaOffense)
                  deltaOffense = ((powerMoments["offenseMoment1"] / (1.-powerMoments["beta1t"]))
                     / (math.sqrt(powerMoments["offenseMoment2"] / (1.-powerMoments["beta2t"])) 
                     + self.gameSimulator.adamParams["epsilon"]))
                  powerMoments["defenseMoment1"] = (self.gameSimulator.adamParams["beta1"]*powerMoments["defenseMoment1"]
                     + (1.0 - self.gameSimulator.adamParams["beta1"])*deltaDefense)
                  powerMoments["defenseMoment2"] = (self.gameSimulator.adamParams["beta2"]*powerMoments["defenseMoment2"]
                     + (1.0 - self.gameSimulator.adamParams["beta2"])*deltaDefense*deltaDefense)
                  deltaDefense = ((powerMoments["defenseMoment1"] / (1.-powerMoments["beta1t"]))
                     / (math.sqrt(powerMoments["defenseMoment2"] / (1.-powerMoments["beta2t"])) 
                     + self.gameSimulator.adamParams["epsilon"]))
                  #print("::: deltaDefense "+str(deltaDefense)+" deltaOffense "+str(deltaOffense)+" deltaDefense0 "+str(deltaDefense0)+" deltaOffense0 "+str(deltaOffense0))
                  teamObject.SetAdamMoments(self.seasonYear, powerMoments)
               offPrev = power["offense"]
               defPrev = power["defense"]
               power["offense"] -= (learningRate * deltaOffense) + powerCostFactor*deltaAvePowerOff
               power["defense"] -= (learningRate * deltaDefense) + powerCostFactor*deltaAvePowerDef
               if (power["offense"] < 0.0):
                  power["offense"] = 0.0
               if (power["defense"] < 0.0):
                  power["defense"] = 0.0
               teamObject.SetPower(self.seasonYear, power["offense"], power["defense"])
               actualPower = teamObject.GetPowerActual(self.seasonYear)
               offDiff = 0.0
               defDiff = 0.0
               if (actualPower["offense"] != None):
                  offDif = (actualPower["offense"] - power["offense"])**2
                  defDif = (actualPower["defense"] - power["defense"])**2
                  offensePowerDiff += (offDif)**2
                  defensePowerDiff += (defDif)**2
               if (verbose):
                  print("::: team "+teamName+" season "+str(self.seasonYear)
                     +" actual off="+str(actualPower["offense"])+" def="+str(actualPower["defense"])
                     +" prev off="+str(offPrev)+" def="+str(defPrev)
                     +" dOffDs="+str(teamPowerDeltas[teamObject.teamName]["offenseDs"]/teamPowerDeltas[teamObject.teamName]["count"])
                     +" dOffDt="+str(teamPowerDeltas[teamObject.teamName]["offenseDt"]/teamPowerDeltas[teamObject.teamName]["count"])
                     +" dOff="+str(-(learningRate * deltaOffense))+"+"+str(-powerCostFactor*deltaAvePowerOff)
                     +" dDef="+str(-(learningRate * deltaDefense))+"+"+str(-powerCostFactor*deltaAvePowerDef)
                     +" new off="+str(power["offense"])+" def="+str(power["defense"])
                     +" diffSq off="+str((offDiff)**2)+" def="+str((defDiff)**2))
            offensePowerDiff /= len(teamPowerDeltas)
            defensePowerDiff /= len(teamPowerDeltas)
            if (verbose):
               print("RMS error in actual vs derived power: offense "+str(math.sqrt(offensePowerDiff))+" defense "+str(math.sqrt(defensePowerDiff)))
           
         costUpdated = 0
         countUpdated = 0
         for iRound in range(roundStopCount):
            if (self.seasonInfo.get(self.roundNames[iRound]) != None):
               for game in self.seasonInfo[self.roundNames[iRound]]:
                  power1 = game.team1Object.GetPower(self.seasonYear)
                  #power1a = game.team1Object.GetPower(self.seasonYear-1)
                  power2 = game.team2Object.GetPower(self.seasonYear)
                  #power2a = game.team2Object.GetPower(self.seasonYear-1)
                  #print(":::-2 team1Name "+game.team1Object.teamName+" prevYr "+str(power1a)+" cur "+str(power1))
                  #print(":::-2 team2Name "+game.team2Object.teamName+" prevYr "+str(power2a)+" cur "+str(power2))
                  scores = self.gameSimulator.GetScores(
                           power1["offense"], power1["defense"], 
                           power2["offense"], power2["defense"], game.homeField, (costUpdated == 0))
                  #if (parameters["teamPowers"] and costUpdated == 0 and verbose):
                  #   actualPower1 = game.team1Object.GetPowerActual(self.seasonYear)
                  #   actualPower2 = game.team2Object.GetPowerActual(self.seasonYear)
                  #   if (actualPower1 == None or actualPower1["offense"] == None):
                  #      actualPower1 = power1
                  #      print("::: actualPower1 not available")
                  #   if (actualPower2 == None or actualPower2["offense"] == None):
                  #      actualPower2 = power2
                  #      print("::: actualPower2 not available")
                  #   # print out first game as a sample
                  #   print("::: costUpdated "+str(scores)+" actual "+str(game.score1)+" "+str(game.score2)+" "+str(power1)+" "+str(power2)+" act-cur"+" dd1 "+str(actualPower1["defense"]-power1["defense"])+" do1 "+str(actualPower1["offense"]-power1["offense"])+" dd2 "+str(actualPower2["defense"]-power2["defense"])+" do2 "+str(actualPower2["offense"]-power2["offense"]))
                  countUpdated += 1
                  if (parameters["deprioritizeTotalForCost"] == 0):
                     costUpdated += (scores["score1"] - game.score1)**2 + (scores["score2"] - game.score2)**2
                  else:
                     costUpdated += ((scores["score1"] - scores["score2"] - game.score1 + game.score2)**2
                           + parameters["deprioritizeTotalForCost"]*(scores["score1"] + scores["score2"] - game.score1 - game.score2)**2)
                  #print("::: costUpdated in progress "+str(costUpdated))
                     #print(":::-2 "+game.team1Object.teamName+"-"+game.team2Object.teamName+" "+str(costUpdated)+" s1 "+str(scores["score1"])+" s2 "+str(scores["score2"])+" g1 "+str(game.score1)+" g2 "+str(game.score2))
           
         if (verbose):
            print("AdjustPowerAndLinearParametersFromScores: cost reduction="
                    +str((costOrig-costUpdated)/linearDeltas["count"])
                    +" current cost="+str(costUpdated/linearDeltas["count"]))
         #print("::: flags: "+str(parameters))
         if (parameters["linearParameters"]):
            print("::: current"
               +" spdCoefOff="+str(self.gameSimulator.spdCoefOff)
               +" spdCoefDef="+str(self.gameSimulator.spdCoefDef)
               +" totCoefOff="+str(self.gameSimulator.totCoefOff)
               +" totCoefDef="+str(self.gameSimulator.totCoefDef)
               +" totCoef0="+str(self.gameSimulator.totCoef0))
         if (parameters["homeFieldAdvantage"]):
            print("::: current homeFieldAdvantage="+str(self.gameSimulator.homeFieldAdvantage))

         costInfo = {}
         costInfo["costOrig"] = costOrig/linearDeltas["count"]
         costInfo["costUpdated"] = costUpdated/linearDeltas["count"]
         costInfo["count"] = linearDeltas["count"]
      else: # count is zero
         costInfo = {}
         costInfo["costOrig"] = 0.0
         costInfo["costUpdated"] = 0.0
         costInfo["count"] = linearDeltas["count"]

      return costInfo
              
GameSeason.GetDefaultPower = staticmethod(GameSeason.GetDefaultPower)


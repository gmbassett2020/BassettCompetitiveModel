
#
# Team class
#

class Team:
   """Class holding team information"""

   nextTeamId = 0
    
   teamObjectById = {}
   teamObjectByName = {}
    
   # inactiveName = "INACTIVE" # If this string is returned for conference, division or team name then that team is currently inactive.
    
   @staticmethod
   def IsTeamActiveById(teamId, year):
      if (teamId >= Team.nextTeamId):
         return False
      if (Team.teamObjectById[teamId].schedule.get(year) == None):
         return False
      return True
    
   @staticmethod
   def GetListOfTeamIds(year):
      listOfTeamIds = []
      for teamId in range(Team.nextTeamId):
         if (Team.IsTeamActiveById(teamId, year)):
            listOfTeamIds.append(teamId)
      return listOfTeamIds

   # Note: if a team does not play for a given year, set conferenceName and divisionName to "unassigned".
    
   def __init__(self, teamName, startYear=None, conferenceName=None, divisionName=None):
      self.teamName = teamName
      self.teamId = Team.nextTeamId
      Team.teamObjectById[self.teamId] = self
      Team.teamObjectByName[self.teamName] = self
      Team.nextTeamId += 1
      self.conferenceName = {}
      self.divisionName = {}
      self.defensePower = {}
      self.offensePower = {}
      self.defensePowerActual = {} # For simulated seasons, the "actual" power for the given year
      self.offensePowerActual = {} # For simulated seasons, the "actual" power for the given year
      self.fitPowerPerYearPerRound = {}
      self.schedule = {}
      self.homeField = {} # 1-home, 0.5-neutral, 0-away
      self.startYear = startYear
      self.adamMoments = {} # Power moments for Adam: A method for stochastic optimization
      if (startYear != None):
         if (conferenceName != None):
            self.conferenceName[startYear] = conferenceName
         if (divisionName != None):
            self.divisionName[startYear] = divisionName

   def SetConferenceAndDivision(self, year, conference, division):
      self.conferenceName[year] = conference
      self.divisionName[year] = division

   def GetConferenceName(self, year):
      yearCheck = year
      if (self.startYear == None):
         return self.conferenceName.get(year)
      else:
         while(self.conferenceName.get(yearCheck) == None and yearCheck > self.startYear):
            yearCheck = yearCheck - 1
         return self.conferenceName.get(yearCheck)
 
   def GetDivisionName(self, year):
      yearCheck = year
      if (self.startYear == None):
         return self.divisionName.get(year)
      else:
         while(self.divisionName.get(yearCheck) == None and yearCheck > self.startYear):
            yearCheck = yearCheck - 1
         return self.divisionName.get(yearCheck)

   def SetPower(self, year, offensePower, defensePower):
      # Current power for a given season using best estimate for scores through latest week.
      #print("::: SetPower for "+self.teamName+" id "+str(self.teamId)+" year "+str(year)+" off "+str(offensePower)+" def "+str(defensePower))
      self.defensePower[year] = defensePower
      self.offensePower[year] = offensePower

   def GetOffensePower(self, year):
      return self.offensePower[year]
   
   def GetDefensePower(self, year):
      return self.defensePower[year]
    
   def GetPower(self, year):
      power = {}
      #print("::: GetPower for "+self.teamName+" id "+str(self.teamId)+" year "+str(year))
      power["defense"] = self.defensePower.get(year)
      power["offense"] = self.offensePower.get(year)
      return power
        
   def SetPowerActual(self, year, offensePower, defensePower):
      # Actual power is the real truth power for a simulated season.  It is used when testing which weights to use to get current power for a given week.
      self.defensePowerActual[year] = defensePower
      self.offensePowerActual[year] = offensePower

   def GetAdamMoments(self, year, updateBetaTs=True):
      moments = {}
      if (self.adamMoments.get(year) == None):
         moments["offenseMoment1"] = 0
         moments["offenseMoment2"] = 0
         moments["defenseMoment1"] = 0
         moments["defenseMoment2"] = 0
         moments["beta1t"] = 1.0
         moments["beta2t"] = 1.0
         self.adamMoments[year] = moments
      else:
         moments = self.adamMoments[year]
         #moments["offenseMoment1"] = self.adamMoments[year]["offenseMoment1"]
         #moments["offenseMoment2"] = self.adamMoments[year]["offenseMoment2"]
         #moments["defenseMoment1"] = self.adamMoments[year]["defenseMoment1"]
         #moments["defenseMoment2"] = self.adamMoments[year]["defenseMoment2"]
      return moments

   def SetAdamMoments(self, year, moments):
      self.adamMoments[year] = moments

   def GetPowerActual(self, year):
      # Actual power is the real truth power for a simulated season.  It is used when testing which weights to use to get current power for a given week.
      actualPower = {}
      actualPower["offense"] = self.offensePowerActual.get(year)
      actualPower["defense"] = self.defensePowerActual.get(year)
      return actualPower
    
   def SetFitPower(self, year, roundName, offense, defense):
      # This is the best estimate for team power after averaging fit power for all rounds up through the given round.
      if (self.fitPowerPerYearPerRound.get(year) == None):
         self.fitPowerPerYearPerRound[year] = {}
      if (self.fitPowerPerYearPerRound[year].get(roundName) == None):
         self.fitPowerPerYearPerRound[year][roundName] = {}
      self.fitPowerPerYearPerRound[year][roundName]["offense"] = offense
      self.fitPowerPerYearPerRound[year][roundName]["defense"] = defense   
        
   def GetFitPower(self, year, roundName):
      # This is the best estimate for team power after averaging fit power for all rounds up through the given round.
      if (self.fitPowerPerYearPerRound.get(year) == None 
         or self.fitPowerPerYearPerRound[year].get(roundName) == None):
         return None
         #print("::: fit power for "+str(year)+" "+roundName+" not found.  Returning general power values.")
         #power = {}
         #power["offense"] = self.offensePower[year]
         #power["defense"] = self.defensePower[year]
         #return power
      else:
         return self.fitPowerPerYearPerRound[year][roundName]     
        
   def SetOpponent(self, year, roundName, opponentName, homeField):
      if (self.schedule.get(year) == None):
         self.schedule[year] = {}
      self.schedule[year][roundName] = opponentName
      if (self.homeField.get(year) == None):
         self.homeField[year] = {}
      self.homeField[year][roundName] = homeField

   def GetOpponent(self, year, roundName):
      if (self.schedule.get(year) == None):
         return None
      return self.schedule[year].get(roundName)

   def GetHomeField(self, year, roundName):
      if (self.homeField.get(year) == None):
         return None
      return self.schedule[year].get(roundName)
        

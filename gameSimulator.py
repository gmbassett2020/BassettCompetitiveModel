
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

#
# Game simulator
#

class GameSimulator:
   """Class for generating results/predictions of games"""

   def __init__(self, homeFieldAdvantage, spdCoefOff, spdCoefDef, totCoef0, totCoefOff, totCoefDef, roundScores, spreadRms, maxConfGameWeights):
      self.homeFieldAdvantage = homeFieldAdvantage
      self.spdCoefOff = spdCoefOff
      self.spdCoefDef = spdCoefDef
      self.totCoef0 = totCoef0
      self.totCoefOff = totCoefOff
      self.totCoefDef = totCoefDef
      self.beta = totCoefDef/totCoefOff
      self.roundScores = roundScores
      self.spreadRms = spreadRms
      # First implementation of weights, having different weights per game type.
      self.weights = {}
      self.deltaWeights = {}
      self.weights["conference"] = []
      self.weights["nonconference"] = []
      self.weights["nondivision"] = []
      self.weights["previousSeason"] = []
      self.deltaWeights["conference"] = []
      self.deltaWeights["nonconference"] = []
      self.deltaWeights["nondivision"] = []
      self.deltaWeights["previousSeason"] = []
      self.maxConferenceGameWeights = maxConfGameWeights
      self.maxWeekWeightCount = 20
      # Simplified weights
      self.weekWeights = []
      self.adamMoments = {}
      self.adamMoments["totCoef0"] = {}
      self.adamMoments["totCoefOff"] = {}
      self.adamMoments["totCoefDef"] = {}
      self.adamMoments["spdCoefOff"] = {}
      self.adamMoments["spdCoefDef"] = {}
      self.adamMoments["homeFieldAdvantage"] = {}
      self.adamParams = {}
      self.adamParams["beta1"] = 0.9
      self.adamParams["beta2"] = 0.999
      self.adamParams["epsilon"] = 1e-8
      self.adamParams["beta1t"] = 1.0
      self.adamParams["beta2t"] = 1.0
      for coef in self.adamMoments.keys():
         self.adamMoments[coef]["m"] = 0
         self.adamMoments[coef]["v"] = 0
      while (len(self.weights["conference"]) < self.maxConferenceGameWeights):
         self.weights["conference"].append(1.)
         self.weights["nonconference"].append(1.)
         self.weights["nondivision"].append(1.)
         self.weights["previousSeason"].append(1.)
         self.deltaWeights["conference"].append(0.)
         self.deltaWeights["nonconference"].append(0.)
         self.deltaWeights["nondivision"].append(0.)
         self.deltaWeights["previousSeason"].append(0.)
      while (len(self.weekWeights) < self.maxWeekWeightCount):
         self.weekWeights.append(1.)
    
   def RandomizeWeights(self):
      print ("::: RandomizeWeights called")
      for iCount in range(len(self.weights["conference"])): 
         self.weights["conference"][iCount] = random.random()
         self.weights["nondivision"][iCount] = random.random()
         self.weights["nonconference"][iCount] = random.random()
         self.weights["previousSeason"][iCount] = random.random()

   def SetWeightsToBestFit(self):
      # see simFb-8_normalize.py
      bestWeights = {
         'conference':     [0.06948458898871437, 0.199425108516972, 0.4300651159898038, 0.565155359458553, 0.6184868516256364, 
                            0.6435493056115315, 0.6599233766897128, 0.4308810760326026, 0.4308810760326026, 0.4308810760326026], 
         'nonconference':  [0.3029524508474348, 0.3877423058498223, 0.2750799016569184, 0.19584673049348283, 0.17575458286263607, 
                            0.1755334014770109, 0.15068290958072475, 0.2780482767767577, 0.2780482767767577, 0.2780482767767577], 
         'nondivision':    [0.21385467336661948, 0.17744696454415482, 0.11094779139734445, 0.08353712224844455, 0.0757286473860579, 
                            0.055009713583057326, 0.0855722798119274, 0.12410602700849183, 0.12410602700849183, 0.12410602700849183], 
         'previousSeason': [0.4137082867972314, 0.2353856210890509, 0.18390719095593328, 0.15546078779951966, 0.1300299181256695, 
                            0.1259075793284003, 0.10382143391763506, 0.16696462018214794, 0.16696462018214794, 0.16696462018214794]}

      for iCount in range(len(self.weights["conference"])): 
         self.weights["conference"][iCount] = bestWeights["conference"][iCount]
         self.weights["nondivision"][iCount] = bestWeights["nondivision"][iCount]
         self.weights["nonconference"][iCount] = bestWeights["nonconference"][iCount]
         self.weights["previousSeason"][iCount] = bestWeights["previousSeason"][iCount]

      print("Setting weights to "+str(self.weights))

   def SetWeekWeightsToBestFit(self):
      #2023-08-21
      #Found optimized weights using the following:
      #numberOfSeasons = 100
      #numberOfIterationsForEachWeek = 300
      #numberOfIterationsForWeekWeights = 300
      # iWeekWeightIter 299 cost [2.540729170488716, 3.2494912466686587, 2.9159332983007564, 2.492151096500357, 2.078530999979104, 1.6805707654898485, 1.3180204509279119, 0.9948134722125765, 0.7095372492187236, 0.4895885685340341, 0.31366504326693634, 0.23683016621781747, 0.18779311163478776, 0, 0, 0, 0, 0, 0, 0]
      # iWeekWeightIter 299 weight [0.8140645979562876, 0.6598250349845416, 0.6760580664368508, 0.7159214887274042, 0.7564336711722774, 0.7984806907197232, 0.8401389606679754, 0.8776309277658789, 0.910508970314625, 0.9389177480378471, 0.9601690539204841, 0.9689983845809068, 0.9903137951598799, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      #bestWeights = [0.8140645979562876, 0.6598250349845416, 0.6760580664368508, 0.7159214887274042, 0.7564336711722774, 0.7984806907197232, 0.8401389606679754, 0.8776309277658789, 0.910508970314625, 0.9389177480378471, 0.9601690539204841, 0.9689983845809068, 0.9903137951598799, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      #2023-08-22
      #New weights after removing rounding from predictions and adjusting initialization
      # iWeekWeightIter 299 cost [6.061021605744927, 4.721845012974696, 3.643352469778301, 2.8629141073363282, 2.25540330871929, 1.74958077168523, 1.3309976628132625, 0.9841611102952907, 0.6954937947594652, 0.47059225292020385, 0.2938697212488857, 0.21901620500612098, 0.17622912277768846, 0, 0, 0, 0, 0, 0, 0]
      # iWeekWeightIter 299 weight [0.7941167790817569, 0.7195257962111826, 0.7574704435994234, 0.7948800055001095, 0.8262671256461386, 0.8557465042369405, 0.8813065655037028, 0.9058579695882067, 0.928092095456229, 0.9493606398415331, 0.9662703181070563, 0.9720919860088111, 0.9938275247806949, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
      #3190.843u 0.326s 53:11.38 99.9% 0+0k 0+14048io 0pf+0w
      bestWeights = [0.7941167790817569, 0.7195257962111826, 0.7574704435994234, 0.7948800055001095, 0.8262671256461386, 0.8557465042369405, 0.8813065655037028, 0.9058579695882067, 0.928092095456229, 0.9493606398415331, 0.9662703181070563, 0.9720919860088111, 0.9938275247806949, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

      for iCount in range(len(self.weekWeights)): 
         self.weekWeights[iCount] = bestWeights[iCount]

      print("Setting week weights to "+str(self.weekWeights))

   def SetWeights(self, value=1):
      for iCount in range(len(self.weights["conference"])): 
         self.weights["conference"][iCount] = value
         self.weights["nondivision"][iCount] = value
         self.weights["nonconference"][iCount] = value
         self.weights["previousSeason"][iCount] = value

   def ResetDeltaWeights(self):
      for iCount in range(len(self.weights["conference"])): 
         self.deltaWeights["conference"][iCount] = 0
         self.deltaWeights["nondivision"][iCount] = 0
         self.deltaWeights["nonconference"][iCount] = 0
         self.deltaWeights["previousSeason"][iCount] = 0

   def ApplyDeltaWeights(self):
      for iCount in range(len(self.weights["conference"])): 
         self.weights["conference"][iCount] += self.deltaWeights["conference"][iCount]
         self.weights["nondivision"][iCount] += self.deltaWeights["nondivision"][iCount]
         self.weights["nonconference"][iCount] += self.deltaWeights["nonconference"][iCount]
         self.weights["previousSeason"][iCount] += self.deltaWeights["previousSeason"][iCount]

   def RandomizeLinearParameters(self, randomizeHomeFieldAdvantage=True, randomizeLinearParameters=True):
      print ("::: RandomizeLinearParameters called")
      # Reset t=0
      #self.adamParams["beta1t"] = 1.0
      #self.adamParams["beta2t"] = 1.0
      #self.fitExpectedToBeStable = False
      if (randomizeHomeFieldAdvantage):
         self.homeFieldAdvantage = 5*random.random()
      if (randomizeLinearParameters):
         self.totCoef0 = 20*random.random()
         self.totCoefOff = 10*random.random()
         self.totCoefDef = -10*random.random()
         self.spdCoefOff = 10*random.random()
         self.spdCoefDef = 10*random.random()
    
   def GetScoreAndProbability(self, o1, d1, o2, d2, homeField, dO1=None, dD1=None, dO2=None, dD2=None):
      # Return the average score and probability expected for the given power and home field.

      results = self.GetScores(o1, d1, o2, d2, homeField)
      results["scoreRms"] = self.spreadRms # rmsFactor
      spread = results["score1"] - results["score2"]
      prob = 0.5*(1+special.erf(spread/self.spreadRms/math.sqrt(2)))
      results["probability"] = prob

      # Estimate the variation in the spread from the uncertainty in the powers.
      if (dO1 != None):
         resultsEach = []
         spreadAverage = 0
         probAverage = 0
         for iO1 in range(-1,2,1):
            thisO1 = o1 + iO1*dO1
            for iD1 in range(-1,2,1):
               thisD1 = d1 + iD1*dD1
               for iO2 in range(-1,2,1):
                  thisO2 = o2 + iO2*dO2
                  for iD2 in range(-1,2,1):
                     thisD2 = d2 + iD2*dD2
                     thisResult = self.GetScores(thisO1, thisD1, thisO2, thisD2, homeField)
                     resultsEach.append(thisResult)
                     spreadAverage += thisResult["score1"] - thisResult["score2"]
         spreadAverage /= len(resultsEach)
         spreadRms = 0
         for iEach in range(len(resultsEach)):
            spreadDiff = spreadAverage - (resultsEach[iEach]["score2"] - resultsEach[iEach]["score1"])
            spreadRms += spreadDiff*spreadDiff
         spreadRms = math.sqrt(spreadRms/len(resultsEach))
         totalSpreadRms = math.sqrt(self.spreadRms*self.spreadRms+spreadRms*spreadRms)
         results["scoreRms"] = totalSpreadRms
         prob2 = 0.5*(1+special.erf(spread/totalSpreadRms/math.sqrt(2)))
         results["probability"] = prob2

      #rms = self.rmsCoef0 + (o1+o2)*self.rmsCoefOff + (d1+d2)*self.rmsCoefDef
      # See checkSpreadVsProb.csv which has several years of predictions from 65-75% and average spread is (was 13.4 if only couting wins but dropped to 7 after counting all games forcast to average 70% win) 7.
      # prob = 0.5*(1+special.erf(13.4/25/math.sqrt(2)))
      # print("prob "+str(prob))
      # prob 0.7040207247756906
      # => rmsFactor = 13 to 25
      # lets go with 14 as that is what the theoretical spread RMS has been.
      #rms = self.rmsCoef0 + (o1+o2)*self.rmsCoefOff + (d1+d2)*self.rmsCoefDef
      #rmsFactor = 14

      return results
    
   def GetScores(self, o1, d1, o2, d2, homeField, verbose=False):
      # Return the average score and probability expected for the given power and home field.
      #print("::: off1 "+str(o1)+" def1 "+str(d1)+" off2 "+str(o2)+" def2 "+str(d2))
      results = {}
      total = self.totCoefOff*(o1+o2) + self.totCoefDef*(d1+d2) + self.totCoef0
      spread = self.spdCoefOff*(o1-o2) + self.spdCoefDef*(d1-d2)
      if (spread > 0.0):
         s1 = spread + 0.5*total
         s2 = 0.5*total
      else:
         s2 = -spread + 0.5*total
         s1 = 0.5*total
      #if (verbose):
      #   print("::: s1 "+str(s1)+" coef0 "+str(self.totCoef0))
      if (homeField == 1.):
         s1 += self.homeFieldAdvantage/2.
         s2 -= self.homeFieldAdvantage/2.
      if (homeField == 0.):
         s1 -= self.homeFieldAdvantage/2.
         s2 += self.homeFieldAdvantage/2.
      if (s1 < 0.):
         s1 = 0.
      if (s2 < 0.):
         s2 = 0.
      if (self.roundScores):
         s1 = int(s1+0.5)
         s2 = int(s2+0.5)

      #if (s1 < 0):
      #   s1 = 0.
      #if (s2 < 0):
      #   s2 = 0.
      results["score1"] = s1
      results["score2"] = s2
      return results

   def GetTotalNoSpread(self, o1, d1, o2, d2, actualScore1, actualScore2, homeField, verbose=False):
      # Return the average score and probability expected for the given power and home field.
      results = {}
      total = self.totCoefOff*(o1+o2) + self.totCoefDef*(d1+d2) + self.totCoef0

      # Remove out home field advantage (opposite sign to the GetScores home field adjustment).
      if (homeField == 1.):
         actualScore1 -= self.homeFieldAdvantage/2.
         actualScore2 += self.homeFieldAdvantage/2.
      if (homeField == 0.):
         actualScore1 += self.homeFieldAdvantage/2.
         actualScore2 -= self.homeFieldAdvantage/2.

      actualTotalNoSpread = 2*min(actualScore1, actualScore2)

      results["totalNoSpread"] = total
      results["actualTotalNoSpread"] = actualTotalNoSpread
      return results

   def GetRandomizedScore(self, o1, d1, o2, d2, homeField):
      averageResults = self.GetScoreAndProbability(o1, d1, o2, d2, homeField)
      scoreRms = averageResults["scoreRms"]
      s1 = scoreRms*special.erfinv(2*random.random()-1.) + averageResults["score1"]
      s2 = scoreRms*special.erfinv(2*random.random()-1.) + averageResults["score2"]
      #print ("::: GetRandomizedScore scores "+str(s1)+"-"+str(s2)+" ave "+str(averageResults["score1"])+"-"+str(averageResults["score2"]))
      # Adjust scores to be non-negagive integers
      if (s1 < 0.):
         s1 = 0.
      if (s2 < 0.):
         s2 = 0.
      # TODO FIXME: "actual" scores should always be an integer - right? since real game scores are always an integer.
      #if (self.roundScores):
      #   s1 = int(s1+0.5)
      #   s2 = int(s2+0.5)
      s1 = int(s1+0.5)
      s2 = int(s2+0.5)
      randomizedScore = {}
      randomizedScore["score1"] = s1
      randomizedScore["score2"] = s2
      randomizedScore["averageScore1"] = averageResults["score1"]
      randomizedScore["averageScore2"] = averageResults["score2"]
      randomizedScore["probability"] = averageResults["probability"]
      return randomizedScore

   def AdjustPowerToFitActualScore(self, o1, d1, o2, d2, homeField, actualScore1, actualScore2):
      # Linear model equations:
      # 
      #    alpha = spd_coef_def/spd_coef_off 
      #    beta = tot_coef_def/tot_coef_off
      # 
      #    spread =  s1 - s2 = spd_coef_off*(o1-o2) + spd_coef_def*(d1-d2)
      #    spread = spd_coef_off*(o1-o2 + alpha*(d1-d2))
      # 
      #    total = s1 + s2 = tot_coef_off*(o1+o2) + tot_coef_off*(d1+d2)
      #    total = s1 + s2 = tot_coef_off*(o1+o2 + beta*(d1+d2))
      # 
      # Equations for adjusting actual vs expected power using actual scores
      # 
      #    Actual spread and total points:
      # 
      #       spread_actual = spd_coef_off*[o1_adjusted-o2_adjusted + alpha*(d1_adjusted-d2_adjusted)]
      #       total_actual = tot_coef_off*[o1_adjusted+o2_adjusted + beta*(d1_adjusted+d2_adjusted)]
      # 
      #    Constraint - total power remains the same:
      # 
      #       o1_orig+alpha*d1_orig + o2_orig+alpha*d2_orig = o1_adjusted+alpha*d1_adjusted + o2_adjusted+alpha*d2_adjusted 
      # 
      #    Constraint - minimize the changes in power:
      # 
      #       changes in power = (o1_orig-o1_adjusted)^2 + alpha^2*(d1_orig-d1_adjusted)^2 + (o2_orig-o2_adjusted)^2 + alpha^2*(d2_orig-d2_adjusted)^2 
      #       d change / d d1_adjusted = 2*[(o1_orig-o1_adjusted)*d o1_adjusted/d d1_adjusted + alpha^2*(d1_orig-d1_adjusted)*d d1_adjusted/d d1_adjusted + (o2_orig-o2_adjusted)*d o2_adjusted/d d1_adjusted + alpha^2*(d2_orig-d2_adjusted)*d d2_adjusted/d d1_adjusted] = 0
      #       (o1_orig-o1_adjusted)*d o1_adjusted/d d1_adjusted + alpha^2*(d1_orig-d1_adjusted)*d d1_adjusted/d d1_adjusted + (o2_orig-o2_adjusted)*d o2_adjusted/d d1_adjusted + alpha^2*(d2_orig-d2_adjusted)*d d2_adjusted/d d1_adjusted = 0
      #
      # Solved equations with maxima: https://home.csulb.edu/~woollett/mbe4solve.pdf
      # See bcm/notes_linear.txt

      adjustedPower = {}
        
      o1Orig = o1
      d1Orig = d1
      o2Orig = o2
      d2Orig = d2
        
      # Fit is done on neutral field.  Remove home field advantage before adjusting.
      if (homeField == 0.):
         actualScore1 += self.homeFieldAdvantage/2.
         actualScore2 -= self.homeFieldAdvantage/2.
      if (homeField == 1.):
         actualScore1 -= self.homeFieldAdvantage/2.
         actualScore2 += self.homeFieldAdvantage/2.
        
      expectedScore = self.GetScoreAndProbability(o1, d1, o2, d2, 0.5)
        
      spreadDelta = expectedScore["score1"] - expectedScore["score2"] - actualScore1 + actualScore2
      totalDelta = expectedScore["score1"] + expectedScore["score2"] - actualScore1 - actualScore2

      alpha = self.spdCoefDef/self.spdCoefOff 
      beta = self.totCoefDef/self.totCoefOff
        
      # o1Actual = o1Orig + o1Delta
      # d1Actual = d1Orig + d1Delta
      # o2Actual = o2Orig + o2Delta
      # d2Actual = d2Orig + d2Delta

      # Spread delta = expected spread - actual spread
      # spreadDelta = spdCoefOff*(o1Delta-o2Delta + alpha*(d1Delta-d2Delta))
        
      # Total delta = expected total - actual total
      # totalDelta = totCoefOff*(o1Delta+o2Delta + beta*(d1Delta+d2Delta))
         
      # Constraint - total power remains the same:
      # o1Delta + alpha*d1Delta + o2Delta + alpha*d2Delta = 0
        
      # Constraint - minimize the change in power:
      # Change in power = o1Delta^2 + (alpha*d1Delta)^2 + o2Delta^2 + (alpha*d2Delta)^2
      # Setting change in power to zero: 
      # o1Delta - o2Delta + alpha*(d2Delta - d1Delta) = 0

      o1Delta = -((2*alpha*self.spdCoefOff*totalDelta+spreadDelta*self.totCoefOff*(alpha-beta))
               /(self.spdCoefOff*4*self.totCoefOff*(alpha-beta)))
      d1Delta = -((spreadDelta*self.totCoefOff*(alpha-beta)-2*alpha*self.spdCoefOff*totalDelta)
               /(self.spdCoefOff*4*alpha*self.totCoefOff*(alpha-beta)))
      o2Delta = ((spreadDelta*self.totCoefOff*(alpha-beta)-2*alpha*self.spdCoefOff*totalDelta)
               /(self.spdCoefOff*4*self.totCoefOff*(alpha-beta)))
      d2Delta = ((2*alpha*self.spdCoefOff*totalDelta+spreadDelta*self.totCoefOff*(alpha-beta))
               /(self.spdCoefOff*4*alpha*self.totCoefOff*(alpha-beta)))

      #print("::: deltas "+str(o1Delta)+" "+str(d1Delta)+" "+str(o2Delta)+" "+str(d2Delta))
      o1Adjusted = o1Orig + o1Delta
      d1Adjusted = d1Orig + d1Delta
      o2Adjusted = o2Orig + o2Delta
      d2Adjusted = d2Orig + d2Delta
        
      adjustedPower["offense1"] = o1Adjusted
      adjustedPower["defense1"] = d1Adjusted
      adjustedPower["offense2"] = o2Adjusted
      adjustedPower["defense2"] = d2Adjusted
        
      if (False):
         # debug info:
         origPower = o1+alpha*d1 + o2+alpha*d2
         newPower = o1Adjusted+alpha*d1Adjusted + o2Adjusted+alpha*d2Adjusted

         verifyAdjusted = self.GetScoreAndProbability(o1Adjusted, d1Adjusted, o2Adjusted, d2Adjusted, 0.5)
         print("::: o&d "+str(o1)+" "+str(d1)+" "+str(o2)+" "+str(d2) 
               +" neutral expected "+str(expectedScore["score1"])+"-"+str(expectedScore["score2"]))
         print("    neutral actual "+str(actualScore1)+" "+str(actualScore2)+" adj o&d "
               +str(o1Adjusted)+" "+str(d1Adjusted)+" "+str(o2Adjusted)+" "+str(d2Adjusted)
               +" verify "+str(verifyAdjusted["score1"])+" "+str(verifyAdjusted["score2"]))
         print("   origPower "+str(origPower)+" adjustedPower "+str(newPower))

      return adjustedPower


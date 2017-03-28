﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class IdentificationData
    {

        public double[][] inputData;
        public double[][] outputData;

        public IdentificationData(double[][] inputData, double[][] outputData)
        {
            this.inputData = inputData;
            this.outputData = outputData;
        }

        #region properties

        public double[][] InputData
        {
            get
            {
                return inputData;
            }
        }

        public int NumInputVariables
        {
            get
            {
                return inputData[0].Length;
            }
        }

        public double[][] OutputData
        {
            get
            {
                return outputData;
            }
        }

        public int NumOutputVariables
        {
            get
            {
                return outputData[0].Length;
            }
        }

        public int NumSamples
        {
            get
            {
                return inputData.Length;
            }
        }

        #endregion properties
    }
}
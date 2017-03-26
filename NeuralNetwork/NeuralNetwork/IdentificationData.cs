using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class IdentificationData
    {

        public double[,] inputData;
        public double[,] outputData;

        public IdentificationData(double[,] inputData, double[,] outputData)
        {
            this.inputData = inputData;
            this.outputData = outputData;
        }

        #region properties

        public double[,] InputData
        {
            get
            {
                return inputData;
            }
        }

        public double NumInputVariables
        {
            get
            {
                return inputData.GetLength(1);
            }
        }

        public double[,] OutputData
        {
            get
            {
                return outputData;
            }
        }

        public double NumOutputVariables
        {
            get
            {
                return outputData.GetLength(1);
            }
        }

        public double NumSamples
        {
            get
            {
                return inputData.GetLength(0);
            }
        }

        #endregion properties
    }
}

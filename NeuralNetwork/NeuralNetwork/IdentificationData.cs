using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    public class IdentificationData
    {

        public double[][] inputData;
        public double[][] outputData;

        public IdentificationData(double[][] inputData)
        {
            this.inputData = inputData;
            this.outputData = new double[inputData.Length][];
        }

        public IdentificationData(double[][] inputData, double[][] outputData) : this(inputData)
        {
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

        #region public methods

        public bool TrySerialize(string filename)
        {
            using (System.IO.StreamWriter file = new FormattingStreamWriter(@filename, System.Globalization.CultureInfo.InvariantCulture))
            {
                for (int i = 0; i < NumSamples; i++)
                {
                    file.WriteLine("{0}\t{1}", InputData[i][0], OutputData[i][0]);
                }
            }

            return false;
        }

        #endregion

    }

    public class FormattingStreamWriter : StreamWriter
    {
        private readonly IFormatProvider formatProvider;

        public FormattingStreamWriter(string path, IFormatProvider formatProvider)
            : base(path)
        {
            this.formatProvider = formatProvider;
        }
        public override IFormatProvider FormatProvider
        {
            get
            {
                return this.formatProvider;
            }
        }
    }
}

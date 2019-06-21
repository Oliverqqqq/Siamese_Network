using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Lucene.Net.Index;
using Lucene.Net.Search;

namespace IFN647_PROJECT
{
    class CustomerSimilarity: DefaultSimilarity
    {
        
        public override float QueryNorm(float sumOfSquaredWeights)
        {
            return 1.0f;
        }

        public override float SloppyFreq(int distance)
        {
            return 1.0f;
        }

        public override float ComputeNorm(string field, FieldInvertState state)
        {
            // using the original formula for calculating the norm
            return base.ComputeNorm(field, state);
        }


        public override float LengthNorm(string fieldName, int numTerms)
            /*
             * 
             * */
        {
            return (float)1.0;
                /// (float)Math.Sqrt(numTerms);
            // return base.LengthNorm(fieldName, numTerms);
        }

        public override float Idf(int docFreq, int numDocs)
        {
            return (float)Math.Log(numDocs / (docFreq + 1)) + 1;
            //return base.Idf(docFreq, numDocs);
        }
        public override float Coord(int overlap, int maxOverlap)
        {
            return (float)overlap / maxOverlap;
            //return base.Coord(overlap, maxOverlap);
        }
        public override float Tf(float freq)
        {
            return (float) Math.Sqrt(freq);
           // return base.Tf(freq);
        }


    }
}

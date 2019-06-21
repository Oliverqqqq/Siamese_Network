using System;

namespace _505_ssignment_01
{
    class Program
    {
        static void Main(string[] args)
        {  //Array of numbers
            int RepeatTime;
            RepeatTime = 10;
            int Min = int.MinValue;
            int Max = int.MaxValue;
            //Array size,can change input size manually
            int[] A = new int[1000];

            Random randNum = new Random();
            for (int i = 0; i < A.Length; i++)
            {
                A[i] = randNum.Next(Min, Max);

            }

            // test Repeat Number 
			int[] Repeat = new int[] { 1, 1, 2, 2, 4, 5, 6, 6, 8, 9 };
			int[] Negative = new int[] { -12, -123, 0, -3, -43, -54, -15, 97, -10, -63 };





            // MinDistance
            int dmin1 = int.MaxValue;
            // Mindstance2
			int dmin2 = int.MaxValue;

			//test Repeat Number
			int dminRepeat1 = int.MaxValue;
			int dminRepeat2 = int.MaxValue;

			//test negative numbers
			int dminNegative1 = int.MaxValue;
			int dminNegative2 = int.MaxValue;
            
            //random data test


            //time the loop
            System.Diagnostics.Stopwatch sw1 = new System.Diagnostics.Stopwatch();

            //time start
            sw1.Start();

            // count the number of loops 

            int count1 = 0;

            //start loop
            for (int t = 0; t < RepeatTime; t++)
            {
				for (int i = 0; i < A.Length; i++)
                {
					for (int j = 0; j < A.Length; j++)
                    {
                        //operation


						if (i != j & (Math.Abs(A[i] - A[j]) < dmin1))




							dmin1 = Math.Abs(A[i] - A[j]);
                        

                        count1++;


                    }
                }
            }
            // time stop
            sw1.Stop();

            //average total time
            long avgtime1 = sw1.ElapsedMilliseconds / RepeatTime;
            //average total operation count
            int avgcount1;
            avgcount1 = count1 / RepeatTime;
            





            //second algorithm
            System.Diagnostics.Stopwatch sw2 = new System.Diagnostics.Stopwatch();

            //time start
            
            sw2.Start();

            //count the numbner of loops

            int count2 = 0;
            // start loop
            for (int t = 0; t < RepeatTime; t++)
            {
				for (int i = 0; i < A.Length - 1; i++)
                {
					for (int j = i + 1; j < A.Length; j++)
                    {       //operation
                        int v = Math.Abs(A[i] - A[j]);
                        
                        
                        
						if (v < dmin2)

							dmin2= v;
                        
                        count2++;
                        


                    }
                }
            }
            //time stop
            sw2.Stop();

            //average total time

            long avgtime2 = sw2.ElapsedMilliseconds / RepeatTime;
            
            //aveage total oepration count
            int avgcount2;
            avgcount2 = count2 / RepeatTime;

            




			// Repeat number test

			//time the loop
            System.Diagnostics.Stopwatch sw3 = new System.Diagnostics.Stopwatch();

            //time start
            sw1.Start();

            // count the number of loops 

            int count3 = 0;

            //start loop
            for (int t = 0; t < RepeatTime; t++)
            {
				for (int i = 0; i < Repeat.Length; i++)
                {
					for (int j = 0; j < Repeat.Length; j++)
                    {
                        //operation


						if (i != j & (Math.Abs(A[i] - A[j]) < dminRepeat1))




							dminRepeat1 = Math.Abs(A[i] - A[j]);


                        count3++;


                    }
                }
            }
            // time stop
            sw3.Stop();

            //average total time
            long avgtime3 = sw3.ElapsedMilliseconds / RepeatTime;
            //average total operation count
            int avgcount3;
            avgcount3 = count3 / RepeatTime;



			//second algorithm
            System.Diagnostics.Stopwatch sw4 = new System.Diagnostics.Stopwatch();

            //time start

            sw4.Start();

            //count the numbner of loops

            int count4 = 0;
            // start loop
            for (int t = 0; t < RepeatTime; t++)
            {
				for (int i = 0; i < Repeat.Length - 1; i++)
                {
					for (int j = i + 1; j < Repeat.Length; j++)
                    {       //operation
                        int v = Math.Abs(A[i] - A[j]);



						if (v < dminRepeat2)

							dminRepeat2 = v;

                        count4++;



                    }
                }
            }
            //time stop
            sw4.Stop();

            //average total time

            long avgtime4 = sw4.ElapsedMilliseconds / RepeatTime;

            //aveage total oepration count
            int avgcount4;
            avgcount4 = count4 / RepeatTime;




            // test Negative number Array


			//time the loop
            System.Diagnostics.Stopwatch sw5 = new System.Diagnostics.Stopwatch();

            //time start
            sw5.Start();

            // count the number of loops 

            int count5 = 0;

            //start loop
            for (int t = 0; t < RepeatTime; t++)
            {
				for (int i = 0; i < Negative.Length; i++)
                {
					for (int j = 0; j < Negative.Length; j++)
                    {
                        //operation


						if (i != j & (Math.Abs(A[i] - A[j]) < dminNegative1))




							dminNegative1 = Math.Abs(A[i] - A[j]);


                        count5++;


                    }
                }
            }
            // time stop
            sw5.Stop();

            //average total time
            long avgtime5 = sw1.ElapsedMilliseconds / RepeatTime;
            //average total operation count
            int avgcount5;
            avgcount5 = count5 / RepeatTime;

			//second algorithm
            System.Diagnostics.Stopwatch sw6 = new System.Diagnostics.Stopwatch();

            //time start
            
            sw6.Start();

            //count the numbner of loops

            int count6 = 0;
            // start loop
            for (int t = 0; t < RepeatTime; t++)
            {
				for (int i = 0; i < Negative.Length - 1; i++)
                {
					for (int j = i + 1; j < Negative.Length; j++)
                    {       //operation
                        int v = Math.Abs(A[i] - A[j]);



						if (v < dminNegative2)

							dminNegative2 = v;

                        count6++;



                    }
                }
            }
            //time stop
            sw6.Stop();

            //average total time

            long avgtime6 = sw6.ElapsedMilliseconds / RepeatTime;

            //aveage total oepration count
            int avgcount6;
            avgcount6 = count6 / RepeatTime;


            

            













			//result
            Console.WriteLine("the first dmin is =" + dmin1);
            Console.WriteLine("the second dmin is =" + dmin2);
            Console.WriteLine("time of first loop=" + avgtime1);
            Console.WriteLine("time of second loop=" + avgtime2);
            Console.WriteLine("the first loop operate =" + avgcount1);
            Console.WriteLine("the second loop operate=" + avgcount2);

			Console.WriteLine("the fisrt dmin in Repeat array is =" + dminRepeat1);       
			Console.WriteLine("the second dmin in Repeat array is  =" + dminRepeat2);
			Console.WriteLine("time of first loop=" + avgtime3);
            Console.WriteLine("time of second loop=" + avgtime4);
            Console.WriteLine("the first loop operate =" + avgcount3);
            Console.WriteLine("the second loop operate=" + avgcount4);

			Console.WriteLine("the first dmin in Nagative Array is =" + dminNegative1);
			Console.WriteLine("the second dmin in Nagative Array is =" + dminNegative2);
			Console.WriteLine("time of first loop=" + avgtime5);
            Console.WriteLine("time of second loop=" + avgtime6);
			Console.WriteLine("the first loop operate =" + avgcount5);
            Console.WriteLine("the second loop operate=" + avgcount6);
            
            

            Console.ReadKey();

            
        }
    }
}




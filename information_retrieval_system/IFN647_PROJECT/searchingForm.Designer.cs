namespace IFN647_PROJECT
{
    partial class searchingForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.searchBox1 = new System.Windows.Forms.TextBox();
            this.searchingbutton1 = new System.Windows.Forms.Button();
            this.dataGridView1 = new System.Windows.Forms.DataGridView();
            this.nextbutton1 = new System.Windows.Forms.Button();
            this.BACKbutton1 = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.viewbutton1 = new System.Windows.Forms.Button();
            this.Savebutton = new System.Windows.Forms.Button();
            this.textBox3 = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.saveFileDialog1 = new System.Windows.Forms.SaveFileDialog();
            this.PhrasecheckBox1 = new System.Windows.Forms.CheckBox();
            this.final_query = new System.Windows.Forms.Label();
            this.querydiaplay = new System.Windows.Forms.TextBox();
            this.synonym = new System.Windows.Forms.CheckBox();
            this.preprocessing = new System.Windows.Forms.CheckBox();
            this.load_infor = new System.Windows.Forms.Button();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.dataGridView2 = new System.Windows.Forms.DataGridView();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView2)).BeginInit();
            this.SuspendLayout();
            // 
            // searchBox1
            // 
            this.searchBox1.Location = new System.Drawing.Point(204, 25);
            this.searchBox1.Name = "searchBox1";
            this.searchBox1.Size = new System.Drawing.Size(955, 25);
            this.searchBox1.TabIndex = 0;
            // 
            // searchingbutton1
            // 
            this.searchingbutton1.BackColor = System.Drawing.Color.Transparent;
            this.searchingbutton1.FlatAppearance.BorderSize = 0;
            this.searchingbutton1.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.searchingbutton1.Image = global::IFN647_PROJECT.Properties.Resources.Search_icon__2_;
            this.searchingbutton1.Location = new System.Drawing.Point(1191, 12);
            this.searchingbutton1.Name = "searchingbutton1";
            this.searchingbutton1.Size = new System.Drawing.Size(106, 64);
            this.searchingbutton1.TabIndex = 1;
            this.searchingbutton1.UseVisualStyleBackColor = false;
            this.searchingbutton1.Click += new System.EventHandler(this.button1_Click);
            // 
            // dataGridView1
            // 
            this.dataGridView1.AutoSizeColumnsMode = System.Windows.Forms.DataGridViewAutoSizeColumnsMode.Fill;
            this.dataGridView1.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dataGridView1.Location = new System.Drawing.Point(870, 130);
            this.dataGridView1.Name = "dataGridView1";
            this.dataGridView1.RowTemplate.Height = 27;
            this.dataGridView1.SelectionMode = System.Windows.Forms.DataGridViewSelectionMode.FullRowSelect;
            this.dataGridView1.Size = new System.Drawing.Size(455, 467);
            this.dataGridView1.TabIndex = 2;
            this.dataGridView1.Click += new System.EventHandler(this.dataGridView1_Click);
            // 
            // nextbutton1
            // 
            this.nextbutton1.Location = new System.Drawing.Point(1217, 613);
            this.nextbutton1.Name = "nextbutton1";
            this.nextbutton1.Size = new System.Drawing.Size(108, 51);
            this.nextbutton1.TabIndex = 3;
            this.nextbutton1.Text = "NEXT";
            this.nextbutton1.UseVisualStyleBackColor = true;
            this.nextbutton1.Click += new System.EventHandler(this.nextbutton1_Click);
            // 
            // BACKbutton1
            // 
            this.BACKbutton1.Location = new System.Drawing.Point(948, 613);
            this.BACKbutton1.Name = "BACKbutton1";
            this.BACKbutton1.Size = new System.Drawing.Size(108, 51);
            this.BACKbutton1.TabIndex = 4;
            this.BACKbutton1.Text = "BACK";
            this.BACKbutton1.UseVisualStyleBackColor = true;
            this.BACKbutton1.Click += new System.EventHandler(this.BACKbutton1_Click);
            // 
            // label1
            // 
            this.label1.Font = new System.Drawing.Font("Arial Narrow", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Location = new System.Drawing.Point(456, 99);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(156, 30);
            this.label1.TabIndex = 5;
            this.label1.Text = "Total Results:";
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(570, 100);
            this.textBox1.Name = "textBox1";
            this.textBox1.ReadOnly = true;
            this.textBox1.Size = new System.Drawing.Size(137, 25);
            this.textBox1.TabIndex = 6;
            // 
            // textBox2
            // 
            this.textBox2.Location = new System.Drawing.Point(897, 99);
            this.textBox2.Name = "textBox2";
            this.textBox2.ReadOnly = true;
            this.textBox2.Size = new System.Drawing.Size(278, 25);
            this.textBox2.TabIndex = 7;
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Font = new System.Drawing.Font("Arial Narrow", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label2.Location = new System.Drawing.Point(794, 100);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(116, 24);
            this.label2.TabIndex = 9;
            this.label2.Text = "Execute Time:";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Font = new System.Drawing.Font("Arial Narrow", 13.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label3.Location = new System.Drawing.Point(1110, 623);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(65, 29);
            this.label3.TabIndex = 10;
            this.label3.Text = "label3";
            // 
            // viewbutton1
            // 
            this.viewbutton1.Location = new System.Drawing.Point(763, 703);
            this.viewbutton1.Name = "viewbutton1";
            this.viewbutton1.Size = new System.Drawing.Size(176, 43);
            this.viewbutton1.TabIndex = 11;
            this.viewbutton1.Text = "SEE entire abstract";
            this.viewbutton1.UseVisualStyleBackColor = true;
            this.viewbutton1.Click += new System.EventHandler(this.viewbutton1_Click_1);
            // 
            // Savebutton
            // 
            this.Savebutton.Location = new System.Drawing.Point(515, 625);
            this.Savebutton.Name = "Savebutton";
            this.Savebutton.Size = new System.Drawing.Size(142, 48);
            this.Savebutton.TabIndex = 12;
            this.Savebutton.Text = "Save Results";
            this.Savebutton.UseVisualStyleBackColor = true;
            this.Savebutton.Click += new System.EventHandler(this.Savebutton_Click);
            // 
            // textBox3
            // 
            this.textBox3.Location = new System.Drawing.Point(170, 639);
            this.textBox3.Name = "textBox3";
            this.textBox3.Size = new System.Drawing.Size(113, 25);
            this.textBox3.TabIndex = 13;
            // 
            // label4
            // 
            this.label4.Font = new System.Drawing.Font("Arial Narrow", 13.8F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label4.Location = new System.Drawing.Point(33, 633);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(131, 38);
            this.label4.TabIndex = 14;
            this.label4.Text = "SET topicID:";
            // 
            // PhrasecheckBox1
            // 
            this.PhrasecheckBox1.AutoSize = true;
            this.PhrasecheckBox1.Font = new System.Drawing.Font("Arial Narrow", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.PhrasecheckBox1.Location = new System.Drawing.Point(27, 136);
            this.PhrasecheckBox1.Name = "PhrasecheckBox1";
            this.PhrasecheckBox1.Size = new System.Drawing.Size(137, 28);
            this.PhrasecheckBox1.TabIndex = 15;
            this.PhrasecheckBox1.Text = "No-preprocess";
            this.PhrasecheckBox1.UseVisualStyleBackColor = true;
            // 
            // final_query
            // 
            this.final_query.AutoSize = true;
            this.final_query.Font = new System.Drawing.Font("Arial Narrow", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.final_query.Location = new System.Drawing.Point(597, 165);
            this.final_query.Name = "final_query";
            this.final_query.Size = new System.Drawing.Size(95, 24);
            this.final_query.TabIndex = 16;
            this.final_query.Text = "Final Quert:";
            // 
            // querydiaplay
            // 
            this.querydiaplay.Location = new System.Drawing.Point(431, 192);
            this.querydiaplay.Multiline = true;
            this.querydiaplay.Name = "querydiaplay";
            this.querydiaplay.Size = new System.Drawing.Size(398, 405);
            this.querydiaplay.TabIndex = 17;
            // 
            // synonym
            // 
            this.synonym.AutoSize = true;
            this.synonym.Font = new System.Drawing.Font("Arial Narrow", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.synonym.Location = new System.Drawing.Point(204, 136);
            this.synonym.Name = "synonym";
            this.synonym.Size = new System.Drawing.Size(100, 28);
            this.synonym.TabIndex = 18;
            this.synonym.Text = "synonym";
            this.synonym.UseVisualStyleBackColor = true;
            // 
            // preprocessing
            // 
            this.preprocessing.AutoSize = true;
            this.preprocessing.Font = new System.Drawing.Font("Arial Narrow", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.preprocessing.Location = new System.Drawing.Point(27, 99);
            this.preprocessing.Name = "preprocessing";
            this.preprocessing.Size = new System.Drawing.Size(132, 28);
            this.preprocessing.TabIndex = 19;
            this.preprocessing.Text = "preprocessing";
            this.preprocessing.UseVisualStyleBackColor = true;
            // 
            // load_infor
            // 
            this.load_infor.Location = new System.Drawing.Point(27, 26);
            this.load_infor.Name = "load_infor";
            this.load_infor.Size = new System.Drawing.Size(142, 67);
            this.load_infor.TabIndex = 20;
            this.load_infor.Text = "load information needs";
            this.load_infor.UseVisualStyleBackColor = true;
            this.load_infor.Click += new System.EventHandler(this.load_infor_Click);
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // dataGridView2
            // 
            this.dataGridView2.AutoSizeColumnsMode = System.Windows.Forms.DataGridViewAutoSizeColumnsMode.Fill;
            this.dataGridView2.ColumnHeadersHeightSizeMode = System.Windows.Forms.DataGridViewColumnHeadersHeightSizeMode.AutoSize;
            this.dataGridView2.Location = new System.Drawing.Point(5, 187);
            this.dataGridView2.Name = "dataGridView2";
            this.dataGridView2.RowTemplate.Height = 27;
            this.dataGridView2.SelectionMode = System.Windows.Forms.DataGridViewSelectionMode.FullRowSelect;
            this.dataGridView2.Size = new System.Drawing.Size(390, 410);
            this.dataGridView2.TabIndex = 21;
            this.dataGridView2.CellContentClick += new System.Windows.Forms.DataGridViewCellEventHandler(this.dataGridView2_CellContentClick);
            // 
            // searchingForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(8F, 15F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.RoyalBlue;
            this.ClientSize = new System.Drawing.Size(1346, 758);
            this.Controls.Add(this.dataGridView2);
            this.Controls.Add(this.load_infor);
            this.Controls.Add(this.preprocessing);
            this.Controls.Add(this.synonym);
            this.Controls.Add(this.querydiaplay);
            this.Controls.Add(this.final_query);
            this.Controls.Add(this.PhrasecheckBox1);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.textBox3);
            this.Controls.Add(this.Savebutton);
            this.Controls.Add(this.viewbutton1);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.textBox2);
            this.Controls.Add(this.textBox1);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.BACKbutton1);
            this.Controls.Add(this.nextbutton1);
            this.Controls.Add(this.dataGridView1);
            this.Controls.Add(this.searchingbutton1);
            this.Controls.Add(this.searchBox1);
            this.Name = "searchingForm";
            this.Text = "searchingForm";
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView1)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.dataGridView2)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox searchBox1;
        private System.Windows.Forms.Button searchingbutton1;
        private System.Windows.Forms.DataGridView dataGridView1;
        private System.Windows.Forms.Button nextbutton1;
        private System.Windows.Forms.Button BACKbutton1;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.TextBox textBox2;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Button viewbutton1;
        private System.Windows.Forms.Button Savebutton;
        private System.Windows.Forms.TextBox textBox3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.SaveFileDialog saveFileDialog1;
        internal System.Windows.Forms.CheckBox PhrasecheckBox1;
        private System.Windows.Forms.Label final_query;
        private System.Windows.Forms.TextBox querydiaplay;
        private System.Windows.Forms.CheckBox synonym;
        private System.Windows.Forms.CheckBox preprocessing;
        private System.Windows.Forms.Button load_infor;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
        private System.Windows.Forms.DataGridView dataGridView2;
    }
}
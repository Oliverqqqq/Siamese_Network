namespace IFN647_PROJECT
{
    partial class indexForm1
    {
        /// <summary>
        /// 必需的设计器变量。
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// 清理所有正在使用的资源。
        /// </summary>
        /// <param name="disposing">如果应释放托管资源，为 true；否则为 false。</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows 窗体设计器生成的代码

        /// <summary>
        /// 设计器支持所需的方法 - 不要修改
        /// 使用代码编辑器修改此方法的内容。
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            this.indexlabel1 = new System.Windows.Forms.Label();
            this.resourcelabel2 = new System.Windows.Forms.Label();
            this.indexBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.resourceBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.collectBox2 = new System.Windows.Forms.TextBox();
            this.indexBox1 = new System.Windows.Forms.TextBox();
            this.SEARCHbutton = new System.Windows.Forms.Button();
            this.conformbutton1 = new System.Windows.Forms.Button();
            this.filebutton2 = new System.Windows.Forms.Button();
            this.indexbutton1 = new System.Windows.Forms.Button();
            this.label1 = new System.Windows.Forms.Label();
            this.helpbutton = new System.Windows.Forms.Button();
            this.contextMenuStrip1 = new System.Windows.Forms.ContextMenuStrip(this.components);
            this.title_boost = new System.Windows.Forms.Label();
            this.Author_boost = new System.Windows.Forms.Label();
            this.title = new System.Windows.Forms.NumericUpDown();
            this.author = new System.Windows.Forms.NumericUpDown();
            this.groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.title)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.author)).BeginInit();
            this.SuspendLayout();
            // 
            // indexlabel1
            // 
            this.indexlabel1.Font = new System.Drawing.Font("Cambria", 10.8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.indexlabel1.ForeColor = System.Drawing.SystemColors.Desktop;
            this.indexlabel1.Location = new System.Drawing.Point(6, 56);
            this.indexlabel1.Name = "indexlabel1";
            this.indexlabel1.Size = new System.Drawing.Size(261, 59);
            this.indexlabel1.TabIndex = 0;
            this.indexlabel1.Text = "please set indexing path:";
            // 
            // resourcelabel2
            // 
            this.resourcelabel2.Font = new System.Drawing.Font("Cambria", 10.8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.resourcelabel2.ForeColor = System.Drawing.SystemColors.Desktop;
            this.resourcelabel2.Location = new System.Drawing.Point(6, 131);
            this.resourcelabel2.Name = "resourcelabel2";
            this.resourcelabel2.Size = new System.Drawing.Size(231, 50);
            this.resourcelabel2.TabIndex = 1;
            this.resourcelabel2.Text = "please set collection path:";
            // 
            // groupBox1
            // 
            this.groupBox1.BackColor = System.Drawing.Color.White;
            this.groupBox1.Controls.Add(this.author);
            this.groupBox1.Controls.Add(this.title);
            this.groupBox1.Controls.Add(this.Author_boost);
            this.groupBox1.Controls.Add(this.title_boost);
            this.groupBox1.Controls.Add(this.collectBox2);
            this.groupBox1.Controls.Add(this.indexBox1);
            this.groupBox1.Controls.Add(this.SEARCHbutton);
            this.groupBox1.Controls.Add(this.conformbutton1);
            this.groupBox1.Controls.Add(this.indexlabel1);
            this.groupBox1.Controls.Add(this.resourcelabel2);
            this.groupBox1.Controls.Add(this.filebutton2);
            this.groupBox1.Controls.Add(this.indexbutton1);
            this.groupBox1.FlatStyle = System.Windows.Forms.FlatStyle.Popup;
            this.groupBox1.Font = new System.Drawing.Font("Cambria", 13.8F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.groupBox1.Location = new System.Drawing.Point(46, 252);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(1043, 435);
            this.groupBox1.TabIndex = 6;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Indexing process";
            // 
            // collectBox2
            // 
            this.collectBox2.Location = new System.Drawing.Point(243, 131);
            this.collectBox2.Name = "collectBox2";
            this.collectBox2.Size = new System.Drawing.Size(316, 34);
            this.collectBox2.TabIndex = 7;
            // 
            // indexBox1
            // 
            this.indexBox1.Location = new System.Drawing.Point(243, 56);
            this.indexBox1.Name = "indexBox1";
            this.indexBox1.Size = new System.Drawing.Size(327, 34);
            this.indexBox1.TabIndex = 6;
            // 
            // SEARCHbutton
            // 
            this.SEARCHbutton.BackColor = System.Drawing.Color.White;
            this.SEARCHbutton.FlatAppearance.BorderSize = 0;
            this.SEARCHbutton.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.SEARCHbutton.Image = global::IFN647_PROJECT.Properties.Resources.Search_icon__1_;
            this.SEARCHbutton.ImageAlign = System.Drawing.ContentAlignment.MiddleLeft;
            this.SEARCHbutton.Location = new System.Drawing.Point(424, 260);
            this.SEARCHbutton.Name = "SEARCHbutton";
            this.SEARCHbutton.Size = new System.Drawing.Size(322, 79);
            this.SEARCHbutton.TabIndex = 5;
            this.SEARCHbutton.Text = "start searching";
            this.SEARCHbutton.UseVisualStyleBackColor = false;
            this.SEARCHbutton.Click += new System.EventHandler(this.SEARCHbutton_Click);
            // 
            // conformbutton1
            // 
            this.conformbutton1.BackColor = System.Drawing.Color.White;
            this.conformbutton1.FlatAppearance.BorderSize = 0;
            this.conformbutton1.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.conformbutton1.Image = global::IFN647_PROJECT.Properties.Resources.Check_2_icon;
            this.conformbutton1.ImageAlign = System.Drawing.ContentAlignment.MiddleLeft;
            this.conformbutton1.Location = new System.Drawing.Point(20, 260);
            this.conformbutton1.Name = "conformbutton1";
            this.conformbutton1.Size = new System.Drawing.Size(350, 79);
            this.conformbutton1.TabIndex = 4;
            this.conformbutton1.Text = "Start Indexing";
            this.conformbutton1.UseVisualStyleBackColor = false;
            this.conformbutton1.Click += new System.EventHandler(this.conformbutton1_Click);
            // 
            // filebutton2
            // 
            this.filebutton2.BackColor = System.Drawing.Color.White;
            this.filebutton2.FlatAppearance.BorderSize = 0;
            this.filebutton2.FlatStyle = System.Windows.Forms.FlatStyle.Popup;
            this.filebutton2.Image = global::IFN647_PROJECT.Properties.Resources.Folder_icon1;
            this.filebutton2.ImageAlign = System.Drawing.ContentAlignment.MiddleLeft;
            this.filebutton2.Location = new System.Drawing.Point(594, 131);
            this.filebutton2.Name = "filebutton2";
            this.filebutton2.Size = new System.Drawing.Size(145, 50);
            this.filebutton2.TabIndex = 3;
            this.filebutton2.Text = "Browse";
            this.filebutton2.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            this.filebutton2.UseVisualStyleBackColor = false;
            this.filebutton2.Click += new System.EventHandler(this.filebutton2_Click);
            // 
            // indexbutton1
            // 
            this.indexbutton1.BackColor = System.Drawing.Color.White;
            this.indexbutton1.FlatAppearance.BorderSize = 0;
            this.indexbutton1.FlatStyle = System.Windows.Forms.FlatStyle.Popup;
            this.indexbutton1.Image = global::IFN647_PROJECT.Properties.Resources.Folder_icon1;
            this.indexbutton1.ImageAlign = System.Drawing.ContentAlignment.MiddleLeft;
            this.indexbutton1.Location = new System.Drawing.Point(594, 54);
            this.indexbutton1.Name = "indexbutton1";
            this.indexbutton1.Size = new System.Drawing.Size(145, 47);
            this.indexbutton1.TabIndex = 2;
            this.indexbutton1.Text = "Browse";
            this.indexbutton1.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            this.indexbutton1.UseVisualStyleBackColor = false;
            this.indexbutton1.Click += new System.EventHandler(this.indexbutton1_Click);
            // 
            // label1
            // 
            this.label1.Font = new System.Drawing.Font("Castellar", 16.2F, ((System.Drawing.FontStyle)((System.Drawing.FontStyle.Bold | System.Drawing.FontStyle.Italic))), System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.Image = global::IFN647_PROJECT.Properties.Resources._240px_QUTLogo_svg;
            this.label1.ImageAlign = System.Drawing.ContentAlignment.TopLeft;
            this.label1.Location = new System.Drawing.Point(40, 43);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(768, 194);
            this.label1.TabIndex = 8;
            this.label1.Text = "EDU INFORMATION SYSTEM";
            this.label1.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // helpbutton
            // 
            this.helpbutton.BackColor = System.Drawing.SystemColors.Info;
            this.helpbutton.ForeColor = System.Drawing.Color.DarkOrchid;
            this.helpbutton.Image = global::IFN647_PROJECT.Properties.Resources.Actions_help_hint_icon;
            this.helpbutton.ImageAlign = System.Drawing.ContentAlignment.TopCenter;
            this.helpbutton.Location = new System.Drawing.Point(1005, 11);
            this.helpbutton.Name = "helpbutton";
            this.helpbutton.Size = new System.Drawing.Size(127, 64);
            this.helpbutton.TabIndex = 5;
            this.helpbutton.Text = "need help?";
            this.helpbutton.TextAlign = System.Drawing.ContentAlignment.BottomCenter;
            this.helpbutton.UseVisualStyleBackColor = false;
            this.helpbutton.Click += new System.EventHandler(this.helpbutton_Click);
            // 
            // contextMenuStrip1
            // 
            this.contextMenuStrip1.ImageScalingSize = new System.Drawing.Size(20, 20);
            this.contextMenuStrip1.Name = "contextMenuStrip1";
            this.contextMenuStrip1.Size = new System.Drawing.Size(61, 4);
            // 
            // title_boost
            // 
            this.title_boost.AutoSize = true;
            this.title_boost.Location = new System.Drawing.Point(806, 30);
            this.title_boost.Name = "title_boost";
            this.title_boost.Size = new System.Drawing.Size(205, 28);
            this.title_boost.TabIndex = 11;
            this.title_boost.Text = "SET TITLE BOOST";
            // 
            // Author_boost
            // 
            this.Author_boost.AutoSize = true;
            this.Author_boost.Location = new System.Drawing.Point(797, 116);
            this.Author_boost.Name = "Author_boost";
            this.Author_boost.Size = new System.Drawing.Size(204, 28);
            this.Author_boost.TabIndex = 12;
            this.Author_boost.Text = "SET Author Boost";
            // 
            // title
            // 
            this.title.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.title.Location = new System.Drawing.Point(839, 79);
            this.title.Name = "title";
            this.title.Size = new System.Drawing.Size(120, 34);
            this.title.TabIndex = 13;
            this.title.Value = new decimal(new int[] {
            10,
            0,
            0,
            65536});
            // 
            // author
            // 
            this.author.Increment = new decimal(new int[] {
            5,
            0,
            0,
            65536});
            this.author.Location = new System.Drawing.Point(839, 161);
            this.author.Name = "author";
            this.author.Size = new System.Drawing.Size(120, 34);
            this.author.TabIndex = 14;
            this.author.Value = new decimal(new int[] {
            1,
            0,
            0,
            0});
            // 
            // indexForm1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 17F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.Color.RoyalBlue;
            this.ClientSize = new System.Drawing.Size(1144, 709);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.groupBox1);
            this.Controls.Add(this.helpbutton);
            this.Font = new System.Drawing.Font("Cambria", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.ForeColor = System.Drawing.SystemColors.Desktop;
            this.HelpButton = true;
            this.Name = "indexForm1";
            this.Text = "PAGE1";
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.title)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.author)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion
        private System.Windows.Forms.Label resourcelabel2;
        private System.Windows.Forms.Button indexbutton1;
        private System.Windows.Forms.Button filebutton2;
        private System.Windows.Forms.FolderBrowserDialog indexBrowserDialog1;
        private System.Windows.Forms.FolderBrowserDialog resourceBrowserDialog1;
        private System.Windows.Forms.Button conformbutton1;
        private System.Windows.Forms.Button helpbutton;
        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label label1;
        public System.Windows.Forms.Label indexlabel1;
        private System.Windows.Forms.Button SEARCHbutton;
        private System.Windows.Forms.TextBox collectBox2;
        private System.Windows.Forms.TextBox indexBox1;
        private System.Windows.Forms.NumericUpDown author;
        private System.Windows.Forms.NumericUpDown title;
        private System.Windows.Forms.Label Author_boost;
        private System.Windows.Forms.Label title_boost;
        private System.Windows.Forms.ContextMenuStrip contextMenuStrip1;
    }
}


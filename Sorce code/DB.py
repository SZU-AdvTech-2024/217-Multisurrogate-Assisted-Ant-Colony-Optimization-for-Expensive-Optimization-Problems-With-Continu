#################################
#        定义类：数据集          #
#################################
class DataBase:
    #初始化类
    def __init__(self, ini_db, Par):
        if isinstance(ini_db,dict):
            self.x_r = ini_db['x_r']
            self.x_c = ini_db['x_c']
            self.f   = ini_db['f']
        else:
            self.x_r = ini_db.x_r
            self.x_c = ini_db.x_c
            self.f   = ini_db.f
                
        self.len_r = numpy.size(self.x_r,1)
        self.len_c = numpy.size(self.x_c,1)
        self.len_f = numpy.size(self.f,1)
        
        self.up_r = Par['up_r']
        self.dn_r = Par['dn_r']
        self.l    = Par['l']
        self.db_size = numpy.size(self.x_r,0)
    
    #函数：更新数据集
    def Update(self,Add_Data):
        self.x_r = numpy.vstack([self.x_r,Add_Data.x_r])
        self.x_c = numpy.vstack([self.x_c,Add_Data.x_c])
        self.f   = numpy.vstack([self.f,Add_Data.f])
        
        self.db_size = numpy.size(self.x_r,0)
        
    #函数：从数据集中选出具有相同离散变量的个体并构建子数据集
    def Same_DB(self,same_c):
        dis = Functional_Operator.Hamming_Dis(self.x_c,same_c)
        idx_same = (dis==0)
        
        nx_r = self.x_r[idx_same,:]
        nx_c = self.x_c[idx_same,:]
        nf   = self.f[idx_same,:]
        
        ini_db = {"x_r":nx_r}
        ini_db.update({"x_c":nx_c})
        ini_db.update({"f":nf})
        
        sub_par = {"up_r":self.up_r}
        sub_par.update({"dn_r":self.dn_r})
        sub_par.update({"l":self.l})
        
        Sub_DB = DataBase(ini_db,sub_par)
        return Sub_DB

    def Converge_curve(self):
        value = []
        FEs_list = []
        for i in range(0,self.db_size):
            value.append(numpy.min(self.f[:i+1,0]))
            FEs_list.append(i+1)
        return value,FEs_list
            
#################################
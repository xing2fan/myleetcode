#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>

#include <stack>
#include <algorithm>
#include <cstring>

using namespace std;

/*
字符串

1.需要的概念：
回文
子串（连续）
子序列（不连续）
前缀树（Trie 树）
后缀树和后缀数组
匹配
字典序
2.需要的操作：
与数组有关的操作：增删改查
字符替换
字符串旋转

3.int 和 long 所能表达的整数范围有限，所以常会使用字符串实现大整数
与大整数相关的加减乘除操作，需要模拟笔算的过程。

4.最长公共子串、最长公共子序列、最长回文子串、最长回文子序列

*/



//3.给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。
int lengthOfLongestSubstring(string s) 
{

	vector<int> m(256,-1);//256个int的数组模拟哈希表,记录字符出现的位置
	int left = -1;
	int res = 0;
	
	for(int i = 0;i < s.size();i++)
	{
		left = max(left,m[s[i]]);//更新滑动窗口左边界
		m[s[i]] = i;			 //更新右边界
		res = max(res,i - left);//更新滑动窗口大小
	}
	return res;
}


string longestPalindrome(string s) 
{
   int n = s.size();
   vector<vector<int>> dp(n, vector<int>(n));
   string ans;

   //注意：在状态转移方程中，我们是从长度较短的字符串向长度较长的字符串进行转移的，
   //因此一定要注意动态规划的循环顺序。
   for (int l = 0; l < n; ++l) 
   {
	   for (int i = 0; i + l < n; ++i) 
	   {
		   int j = i + l;
		   if (l == 0) 
		   {
			   dp[i][j] = 1;
		   }
		   else if (l == 1)
		   {
			   dp[i][j] = (s[i] == s[j]);
		   }
		   else 
		   {
			   dp[i][j] = (s[i] == s[j] && dp[i + 1][j - 1]);
		   }

		   if (dp[i][j] && l + 1 > ans.size()) 
		   {
			   ans = s.substr(i, l + 1);
		   }
	   }
   }
   return ans;
}
int longestPalindromeSubseq(string s) {
	int n=s.size();
	vector<vector<int>>f(n,vector<int>(n,0));

	for(int i=0; i<n; i++)
	{
		f[i][i]=1;
	}

	//按照当前字符是否相等划分为两个类别：
	//s[i] == s[j] ? f[i][j] = f[i+1][j-1] + 2
	//s[i] != s[j] ? f[i][j] = max(f[i+1][j], f[i][j-1])

	/// ......i......................j........
	//	......i i+1...............j-1.......
	// 0-----len-1
	// 1-----
	for(int len = 2; len <= n; len++)
	{
		for(int i=0; i < n-len+1; i++)
		{
			int j = i + len - 1;
			if(s[i]==s[j])
			{
				f[i][j]=f[i+1][j-1]+2;
			}	 
			else
			{
				f[i][j]=max(f[i+1][j],f[i][j-1]);
			}
		}
	}  
	return f[0][n-1];
}


string convert(string s, int numRows) {

    if (numRows <= 1) 
    {
        return s;
    }
    string res;
    
    int size = 2 * numRows - 2;
    for (int i = 0; i < numRows; ++i) 
    {
        for (int j = i; j < s.size(); j += size) 
        {
            res += s[j];
            int tmp = j + size - 2 * i;
            if (i != 0 && i != numRows - 1 && tmp < s.size()) 
            {
				 res += s[tmp];
			}
        }
    }
    return res;
}


int myAtoi(string str) 
{
   if(str.size() == 0)
   {
	   return 0;
   }
   long res = 0;//要使用long不能是int

   int flag = 1;//默认是正数
   int times = 0;
   int i = 0;

   //step1.jump all space
   while(str[i] != '\0')
   {
	   if(str[i] == ' ')//jump to space
	   {
		   i++;
	   }
	   else
	   {
		   break;
	   }
   }

   //step2.process sign
   if(str[i] == '-')
   {
	   flag = -1;
	   i++;
	   times++;
   }

   if(str[i] == '+')
   {
	   flag = 1;
	   i++;
	   times++;
   }
   //要对-+1这样的数字做出处理
   if(times >= 2)
   {
	   return 0;
   }


   while(str[i] != '\0')
   {
	   if(str[i] >= '0' && str[i] <= '9')
	   {
		   res = 10 * res + flag * (str[i] - '0');
	   }
	   else
	   {
		   return res;//如果当前字符不是数字，就返回之前的值
	   }

	   //要每次都判断res值，今早退出
	   if(res >= INT_MAX)
	   {
		   return INT_MAX;
	   }
	   if( res <= INT_MIN)
	   {
		   return INT_MIN;
	   }


	   i++;
   }
   return res;
}


string intToRoman(int num) {

   //根据题意，4是单独的，9也是单独的，1,4,5,9,10,40,50,90,100,400,500,900,1000,4000...
   //输入确保在 1 到 3999 的范围内故而value最大为1000就可以

   int values[] = {1000,  900,	500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
   string reps[] = {"M", "CM", "D",  "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};

   int len = sizeof(values)/sizeof(values[0]);

   string res;
   for (int i = 0; i < len; i++ )
   { 
	   while(num >= values[i])
	   {
		   res += reps[i];//查表找到罗马数字

		   num -= values[i];//更新num值
	   }
   }
   return res;
}

int getValue(char ch) 
{
    switch(ch) {
        case 'I': return 1;
        case 'V': return 5;
        case 'X': return 10;
        case 'L': return 50;
        case 'C': return 100;
        case 'D': return 500;
        case 'M': return 1000;
        default: return 0;
    }
}

int romanToInt(string s) {
    int sum = 0;
    int preNum = getValue(s[0]);

    for(int i = 1;i < s.size(); i ++) 
    {
        int num = getValue(s[i]);
        if(preNum < num) 
        {
            sum -= preNum;
        } 
        else 
        {
            sum += preNum;
        }
        preNum = num;
    }
    sum += preNum;
    return sum;
}
    

string longestCommonPrefix(vector<string>& strs) {
	if(strs.empty()) 
   	{
	   	return "";//如果容器vecto为空，则返回“”
   	}
   	string res = strs[0];//选择第一个字符串作为对照标准

   	for(int i = 1;i < strs.size();i++)//遍历每个字符串
   	{
	   for(int j = 0;j < res.length();j++)//第一个字符串的长度
	   {
		   if(res[j] == strs[i][j])//遍历对比第一个字符串每个字符
		   {
			   continue;
		   }
		   else
		   {
			   res.erase(j);//找到第一个不符合的字符位置，从pos=j处开始删除直至结尾
			   break;//结束本字符串，继续下一个字符串
		   }
	   }
   }
   return res;
}



//定义一个字符串数组查找表
vector<string> table = {"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

//参数1：输入数组，参数2：输出数组 参数3：某次选择字符串 参数4：index
void traceback(string &digits, vector<string> &res, string str, int index)
{
	if(index == digits.size())
	{
		if(str.size() > 0) 
		{
			res.push_back(str);
		}
		return;
	}

	//将当前输入的数字字符串的i位置的数字字符 映射成查找表中的3个字符组成的字符
	//digits[index] - '2'是当前数字的值 
	string s = table[digits[index] - '2'];
	for(int i = 0; i < s.size(); ++i)
	{
		str.push_back(s[i]);

		traceback(digits,res, str, index + 1);

		str.pop_back();
	}
	
}

vector<string> letterCombinations(string digits) 
{
	//第一步，定义返回结果数组,二维数组所以还需要一个一维中间值string
	vector<string> res;

	//第二步，回溯
	traceback( digits,res, "", 0);

	//第三步返回结果
	return res;
}

bool isValid(string s) {
    int len = s.size();
    if(len == 0)
    {
        return true;
    }
    if(len == 1)
    {
        return false;
    }
    stack<char> temp;
   
    for(int i = 0;i < len;i++)
    {
        if(temp.empty())
        {
             temp.push(s[i]);
        }
        else if(temp.top() == '(' && s[i] == ')')
        {
            temp.pop();
        }
        else if(temp.top() == '[' && s[i] == ']')
        {
            temp.pop();
        }
        else if(temp.top() == '{' && s[i] == '}')
        {
            temp.pop();
        }
        else
        {
            temp.push(s[i]);
        }

    }
    return temp.empty();

}


/*
由于字符串只有左括号和右括号两种字符，而且最终结果必定是左括号n个，右括号n个，
所以我们定义两个变量left和right分别表示:剩余左右括号的个数。

1.如果在某次递归时，剩余左括号的个数大于右括号的个数，说明此时生成的字符串中右括号的个数大于左括号的
个数，即会出现'()('这样的非法串，所以这种情况直接返回，不继续处理。

2.如果left和right都为0，则说明此时生成的字符串已有n个左括号和n个右括号，且字符串合法，则存入结果中
后返回。

3.如果以上两种情况都不满足:
若此时left大于0，则调用递归函数，注意参数的更新，
若此时right大于0，则调用递归函数，同样要更新参数。
*/
void generateParenthesisDFS(int left, int right, string out, vector<string> &res) 
{
   if (left > right)//左括号个数大于有括号数
   {
	   return;
   }
   if (left == 0 && right == 0)//此时需要保存结果返回
   {
	   res.push_back(out);
   }
   else 
   {
	   if (left > 0) 
	   {
		   generateParenthesisDFS(left - 1, right, out + '(', res);
	   }
	   if (right > 0)			 
	   {
		   generateParenthesisDFS(left, right - 1, out + ')', res);
	   }
   }
}

vector<string> generateParenthesis(int n) 
{
	vector<string> res;
	generateParenthesisDFS(n, n, "", res);
	return res;
}

int strStr(string haystack, string needle) 
{
	//kmp算法
	if(needle == "")
	{
		return 0;
	}
	int i = 0;
	int j = 0;
	int res = 0;

	while(haystack[i] != '\0' && needle[j] != '\0')
	{
		if(haystack[i] == needle[j])
		{
			i++;
			j++;
		}
		else
		{
			i = i - j + 1;																	 
			j = 0;//从头开始查找
		}  
	}

	if(needle[j] == '\0')
	{
		return i - j;
	}

	return -1;
}

vector<int> findSubstring(string s, vector<string>& words) 
{
	vector<int> res;
	if(words.size()<1 || s.size()<1 || s.size() < words[0].size()*words.size()) 
	{
        return res;
    }
	int wordLen = words[0].size(), lens = wordLen*words.size();
	int target = 0, cur = 0;
	
	unordered_map<string,int> allWord;
	
	for(auto& it:words)
	{
		allWord[it]++;
		for(auto& i:it) 
        {
            target += i;
        }
	}
	for(int i = 0; i < lens; i++) 
	{
		cur += s[i];
	}
	
	// 先看当前字符串的 ASCII 码相加是否相等 方便去重
	for(int i = 0, j; i<=s.size()-lens; cur -= s[i], cur += s[lens + i++])
	{
		// 快速去重
		if(cur != target) 
        {
            continue;
        }
		// 确认一下，是否为真的匹配
		unordered_map<string,int> tem(allWord);
		for(j=i; j<i+lens; j+=wordLen)
		{
            if(tem[s.substr(j, wordLen)]-- == 0) 
            {
                break;
            }
        }
		if(j == i + lens) 
        {
            res.push_back(i);
        }
	}
	return res;
}

int longestValidParentheses(string s) {
    if (s.size() <= 1) 
    {
        return 0;
    }

    int dp[s.size()];
    memset(dp, 0, sizeof(dp));

    int maxlen = 0;
    for (int i = 0; i < s.size(); ++i) 
    {
        if (i == 0)
        {
            continue;
        }
        //只有为右括号才能扩展dp[i - 1]
        if (s[i] == ')') 
        {
            //第i个字符能与第i?dp[i?1]?1个字符匹配，
            //那么dp[i?1]就能向两边扩展，即有dp[i]=dp[i?1]+2
            if (i - dp[i - 1] - 1 >= 0 && s[i - dp[i - 1] - 1] == '(')
            {
                dp[i] = dp[i - 1] + 2;
            }

            //将当前的dp[i]与dp[i - dp[i]]进行合并组成一个更长的有效括号
            if (i - dp[i] >= 0)
            {
                dp[i] += dp[i - dp[i]];
            }
        }
    }


    //以各个字符为尾的最长有效括号长度，得出字符串的最长有效括号长度
    for (int i = 0; i < s.size(); ++i) 
    {
#if 0
        if (maxlen < dp[i])
        {
            maxlen = dp[i];
        }
#else
        maxlen = max(maxlen,dp[i]);
#endif
    }
    return maxlen;
}



string multiply(string num1, string num2) {
    string res;
	int m = num1.size();
	int n = num2.size();
	int max_len = m + n ;

    //结果最长是m+n位,错位相乘，本来应该少一位，但是最高位可能存在进位，所以结果应该是m+n位。
	vector<int> nums_res(max_len, 0);

    ////模拟手算从最后一位开始处理
    //i:2 1 0
	for (int i = m - 1; i >= 0; i--) 
    {
        //j:2 1 0
		for (int j = n - 1; j >= 0; j--)
        {
            //nums_res[5] nums_res[4] nums_res[3] 
            //nums_res[4] nums_res[3] nums_res[2] 
            //nums_res[3] nums_res[2] nums_res[1]  
			nums_res[i + j + 1] += (num1[i] - '0') * (num2[j] - '0');
		}
	}

    //进位
	for (int i = max_len -1; i>0; i--) 
    {
		nums_res[i - 1] += nums_res[i] / 10;
		nums_res[i] %= 10;
	}


	int i = 0;
    //结果数组num_res，长度是m+n，下标是0~m+n-1，存储的方式和计算一样，采取的从低位到高位的顺序。
    /*对于“0”和“0”输入的加判断*/ 
	while (i < max_len && nums_res[i] == 0)
    {
        i++;
        if(i == max_len)  
        {
            return "0";
        }
    }
        
            
	//转换成字符串
    for (; i < max_len; i++) 
    {
		res += nums_res[i] + '0';
	}
	return res;
}

bool isMatch(string s, string p) {
	int i = 0;
	int j = 0;
	int i_star = -1;
	int j_star = -1;
	int m = s.size();
	int n = p.size();

	while (i < m)
	{
		if (j < n && (s[i] == p[j] || p[j] == '?'))
		{
			++ i, ++ j;// 指针同时往后自增1，表示匹配
		}
		else if (j < n && p[j] == '*') 
		{
			// 记录回溯的位置
			i_star = i;// 记录星号
			j_star = j++;// 并且j移到下一位,准备下个循环s[i]和p[j]的匹配
						 //(也就是匹配0个字符)
		}
		else if (i_star >= 0) 
		{
			// 发现字符不匹配且没有星号出现,但是istar>0 
			// 说明*匹配的字符数可能出错 回溯
			i = ++ i_star;//i回溯到i_star+1，显然匹配字符的量出错，现在多匹配一个，且自身加一
			j = j_star + 1;//j回溯到j_star+1 重新使用p串*后的部分开始对齐s串i_star
		} 
		else 
			return false;
	}
	while (j < n && p[j] == '*') ++ j;// 去除多余星号

	return j == n;
}

int lengthOfLastWord(string s) {
       
    if(s.size() == 0)
    {
        return 0;
    }

    int res = 0;
    for(int i = s.size() - 1;i >= 0;i--)
    {
        if(s[i] != ' ')
        {
            res++;
        }
        else//当前是空格
        {
            if(res != 0)
            {
                break;
            }
        }
    }
    return res;
}


bool isNumber(string s) {
    bool num = false;
    bool numAfterE = false;
    bool dot = false;
    bool exp = false;
    bool sign = false;

    int n = s.size();
    for (int i = 0; i < n; ++i) 
    {
        if (s[i] == ' ')//空格
        {
            //前一个是数字，小数点，指数e，正负号，且当前是空格 且后面不是空格
            if (i < n - 1 && s[i + 1] != ' ' && (num || dot || exp || sign)) 
            {
                return false;
            }
        } 
        else if (s[i] == '+' || s[i] == '-')//正负号
        {
            //当前字符是正负号时，前面一个数字必须是e或者空格才行
            if (i > 0 && s[i - 1] != 'e' && s[i - 1] != ' ') 
            {
                return false;
            }
            sign = true;
        } 
        else if (s[i] >= '0' && s[i] <= '9')//数字 
        {
            num = true;
            numAfterE = true;
        } 
        else if (s[i] == '.')//小数点
        {
            //如果前面有小数点或者有指数e不行，因为e后面的必须是正整数
            if (dot || exp) 
            {
                return false;
            }
            dot = true;
        } 
        else if (s[i] == 'e')//e指数 
        {
            //前面有指数e，或者没有数字，因为e前面必须有数字
            if (exp || !num) 
            {
                return false;
            }
            exp = true;
            numAfterE = false;
        } 
        else 
        {
            return false;
        }
    }
    return num && numAfterE;
}

string addBinary(string a, string b) {
    string res;
    int carry = 0;
    int m = a.size() - 1;
    int n = b.size() - 1;

    int q;
    int p;
    int sum;
    while( m >= 0 || n >= 0 )
    {
        if(m >= 0)
        {
            q = a[m] - '0';
            m--;
        }
        else
        {
            q = 0;
        }

        if(n >= 0)
        {
            p = b[n] -'0';
            n--;
        }
        else
        {
            p = 0;
        }  
        sum = q + p + carry;
        carry = sum >> 1;

        res = to_string(sum % 2) + res; 
    }

    if(1 == carry) 
    {
        res = '1' + res;
    }
    return res;
}


string simplifyPath(string path) {
    //path = "/a/./b/../c/", => "/a/c" 和  path = "/a/./b/c/", => "/a/b/c"
    // a . b .. c ==>/a/c              和  a . b c ===>/a/b/c
    //如果是.不入栈 如果是..需要出栈顶字符串 
    //如果是空的话返回"/"，如果有多个"/"只保留一个。
    //那么我们可以把路径看做是由一个或多个"/"分割开的众多子字符串，把它们分别提取出来一一处理即可
    

    vector<string> v;//存放字符串
    int i = 0;
    while (i < path.size()) 
    {
        //step1.按照//分割字符串
        while (path[i] == '/' && i < path.size()) 
        {
            ++i;//跳过///
        }
        if (i == path.size()) 
        {
            break;
        }

        int start = i;//跳过 / 以后
        while (path[i] != '/' && i < path.size()) 
        {
            ++i;
        }
        int end = i - 1;


        //step2.保存中间值,//如果是.不入栈 如果是..需要出栈顶字符串 
        string s = path.substr(start, end - start + 1);
        if (s == "..")
        {
            if (!v.empty()) 
            {
                v.pop_back(); 
            }
        } 
        else if (s != ".") 
        {
            v.push_back(s);
        }

    }

    //如果栈是空的，直接返回/
    if (v.empty()) 
    {
        return "/";
    }

    //使用 / 重新连接栈中的字符串
    string res;
    for (int i = 0; i < v.size(); ++i) 
    {
        res += ('/' + v[i]);
    }
    return res;
}

int minDistance(string word1, string word2) {
    int n = word1.length();
    int m = word2.length();

    // 有一个字符串为空串
    if (n * m == 0) 
    {
        return n + m;
    }

    // DP 数组
    int D[n + 1][m + 1];

    // 边界状态初始化
    for (int i = 0; i < n + 1; i++) 
    {
        D[i][0] = i;
    }
    for (int j = 0; j < m + 1; j++) 
    {
        D[0][j] = j;
    }

    // 计算所有 DP 值
    for (int i = 1; i < n + 1; i++) 
    {
        for (int j = 1; j < m + 1; j++) 
        {
            int left = D[i - 1][j] + 1;
            int down = D[i][j - 1] + 1;
            int left_down = D[i - 1][j - 1];

            if (word1[i - 1] != word2[j - 1]) 
            {
                left_down += 1;
            }
            D[i][j] = min(left, min(down, left_down));
        }
    }
    return D[n][m];
}


string minWindow(string s, string t) {
	unordered_map<char,int>hash;
	for(auto c : t) 
	{
		hash[c]++;
	}

	int cnt = hash.size();
	string res;
	for(int i = 0 , j = 0, c = 0; i < s.size();i++)
	{
		if(hash[s[i]] == 1) c++;
		hash[s[i]]--;

		while(hash[s[j]] < 0)
		{
			hash[s[j++]]++;
		}
		if(c == cnt)
		{
			if(res.empty() || res.size() > i - j + 1) res = s.substr(j, i - j + 1);
		}
	}
	return res;
}


int numDecodings(string s) {
	if (s.empty() || (s.size() > 1 && s[0] == '0')) 
	{
		return 0;
	}

	vector<int> dp(s.size() + 1, 0);

	dp[0] = 1;

	//当前位为止的解码方法个数=上一步的解码方法个数+上一步独立的个数
	//dp[i] = dp[i-1] + (dp[i-2] if 双字符合格 else 0)
	for (int i = 1; i < dp.size(); ++i) 
	{
		if(s[i-1] == '0')
		{
			dp[i] = 0;
		}
		else
		{
			dp[i] = dp[i-1];
		}

		if (i > 1 && (s[i - 2] == '1' || (s[i - 2] == '2' && s[i - 1] <= '6')))
		{
			dp[i] += dp[i - 2];
		}
	}
	return dp.back();
}


bool isAlphaNum(char &ch) 
{
    if (ch >= 'a' && ch <= 'z') return true;
    if (ch >= 'A' && ch <= 'Z') return true;
    if (ch >= '0' && ch <= '9') return true;
    return false;
}

//验证回文串
//给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。
bool isPalindrome(string s) {

    int left = 0;
    int right = s.size() - 1;

    while (left < right)
    {
        if (!isAlphaNum(s[left])) 
        {
            ++left;
        }
        else if (!isAlphaNum(s[right])) 
        {
            --right;
        }
        //将大写统一转换成小写
        //a的ASCII码数值是97，A的ASCII码数值是65
        else if ((s[left] + 32 - 'a') % 32 != (s[right] + 32 - 'a') % 32) 
        {
            return false;
        }
        else 
        {
            ++left; 
            --right;
        }
    }
    return true;
}

bool isPalindrome1(string s) {
    int start = 0;
    int end = s.length()-1;
	
    while(start < end)
    {
        if(!isalnum(s[start]))
            start++;
        else if(!isalnum(s[end]))
            end--;
        else
            if(tolower(s[start++]) != tolower(s[end--]))
                return false;               
    }
    return true;
}

int compareVersion(string version1, string version2) {
    int n1 = version1.size();
    int n2 = version2.size();

	int i = 0, j = 0, d1 = 0, d2 = 0;
	string v1, v2;

	while (i < n1 || j < n2) 
    {
		while (i < n1 && version1[i] != '.') 
        {
			v1 += (version1[i++]);
		}
        //将C++的string转化为C的字符串数组
        //C库函数 int atoi(const char *str) 把参数 str 所指向的字符串转换为一个整数（类型为 int               // 型）。但不适用于string类串，可以使用string对象中的c_str()函数进行转换。
		d1 = atoi(v1.c_str());
        
		while (j < n2 && version2[j] != '.') 
        {
			v2 += (version2[j++]);
		}
		d2 = atoi(v2.c_str());

		if (d1 > d2)
        {
            return 1;
        }
		else if (d1 < d2) 
        {
            return -1;
        }
		v1.clear(); 
        v2.clear();
		++i; 
        ++j;
	}
	return 0;
}

string compressString(string S) {

    string res;

    int i = 0;
    int j = 0;

    //i用来定位该字母的初始位置，j用来移动
    while(i < S.size())
    {
        while(S[i] == S[j])   
        {
            j++;//右指针
        }
        res += (S[i] + to_string(j - i));//j-i字符出现次数

        i = j;//移动左指针
    }

    //若“压缩”后的字符串没有变短，则返回原先的字符串。
    if(res.size() >= S.size())  
    {
        return S;
    }

    return res;
}


bool oneEditAway(string first, string second) {

	if(first == second)//字符串相等直接返回
	{
		return true;
	}

	const int len1 = first.size();
	const int len2 = second.size();
	if(abs(len1 - len2) > 1)
	{
		return false;
	}

	int i = 0;
	int j = len1 - 1;
	int k = len2 - 1;

	while(i < len1 && i < len2 && first[i] == second[i])
	{ 
		//i从左至右扫描，找到两个字符串相等的最右位置
		++i;
	}

	//两个字符串从后往前扫描
	while(j >= 0 && k >= 0 && first[j] == second[k])
	{ 
		//j、k从右至左扫描
		--j;
		--k;
	}
	return j - i < 1 && k - i < 1;
}


string reverseLeftWords(string s, int n) {
	if(n >= s.size()) 
	{
		return s;
	}

	string ans;
	int cnt = 0;//cnt记录字符个数
	//i的值从n 到  s.size()+n
	for(int i = n; cnt < s.size(); i++)
	{
		ans += s[i % s.size()];
		cnt ++;
	}
	return ans;
}


string reverseWords(string s) {
    reverse(s.begin(), s.end());

    int idx = 0;  //表示非空单词前的空格
    int i = 0, j = 0;  //非空单词的范围

    for(i = 0; i < s.size(); ++ i)
    {
        //针对"   hello world!"
        if(s[i] == ' ')
        {
            continue;  //遇到空格继续后移
        }

        //非开头
        if(idx != 0)
        {
            s[idx++] = ' '; //单词开头补空格
        }

        j = i;  //非空单词的开头
        while(j < s.size() && s[j] != ' ')
        {
            s[idx++] = s[j++];//将非空单词往前移，填满前面的空格
        }

        reverse(s.begin() + idx - (j - i), s.begin() + idx);

        i = j;  //更新i的位置
    }

    
    s.erase(s.begin() + idx, s.end());  //删掉末尾的空格
    return s;
}



//left为左括号数，right为右括号数
void dfs(vector<string> &ans,int n,int left, int right, string now)
{
	if(left == n && left == right)
	{
		ans.push_back(now);
		return;
	}

	if(left < n)
	{
		dfs(ans, n, left + 1, right,now + "("); //放左括
	}
	if(right < n && right < left) 
	{
		dfs(ans, n, left, right + 1,now + ")");//放右括号
	}
}

vector<string> generateParenthesis1(int n) 
{
	vector<string> ans;

	dfs(ans, n, 0, 0,"");

	return ans;
}


string replaceSpaces(string S, int length) {
#if 0
	if (S.empty()) return S;

	//step1.统计空格的个数
	int cnt = 0;
	for (int i = 0; i < length; ++i) 
	{
		if (S[i] == ' ')
		{
			++cnt;
		}
	}

	//step2.计算新字符串的长度
	int newLen = length + cnt * 2;

	//step3.从后往前拷贝
	int i = length - 1;
	int j = newLen - 1;
	for (; i >= 0 && i != j; --i) 
	{
		if (S[i] == ' ') 
		{
			S[j--] = '0';
			S[j--] = '2';
			S[j--] = '%';
		} 
		else 
		{
			S[j--] = S[i];
		}
	}

	S[newLen] = '\0';
	return S;
#else
	//从头遍历，遇到空格加上 %20
	string res;
	for (int i = 0; i < length; i++)
	{
		if (S[i] == ' ') 
		{
			res += "%20";
		}
		else 
		{
			res += S[i];
		}
	}
	return res;
#endif
}

int FirstMeet(string    s)
{
	int countchar[256] = { 0 };
	for (int i = 0; i < s.size(); i++)
	{
		countchar[s[i]]++;
	}
	for (int i = 0; i < s.size(); i++)
	{
		if (countchar[s[i]] == 1)
			return i;
	}
	return -1;
}

int longestCommonSubsequence(string text1, string text2) 
{
	int dp[text1.size()+1][text2.size()+1];
	
	int lenA = text1.length();
	int lenB = text2.length();
	for(int i=0;i<=lenA;i++)
	{
		dp[i][0] = 0;
	}
	for(int j=0;j<=lenB;j++)
	{
		dp[0][j] = 0;
	}
	for(int i = 1;i <= lenA;i++)
	{
		 for(int j = 1;j <= lenB;j++)
		 {
			if(text2[j-1] == text1[i-1])
			{
				dp[i][j] = dp[i-1][j-1] + 1;
			}
			else
			{
				dp[i][j] = max(dp[i-1][j],dp[i][j-1]);
			}
		}
	}
	return dp[lenA][lenB];
}



void reverseStr(string& str) 
{ 
    int n = str.length(); 
    for (int i = 0; i < n / 2; i++) 
    {
        swap(str[i], str[n - i - 1]); 
    }
} 


int main(int argc,char **argv)
{
	string input = "1234567abcdefg";

    int res = myAtoi(input);

    string s1 = "abcdefg"; 
    cout << "s1 = "<< s1 << endl;  

    string s2 = "we are a student.";

    string ans;
    ans = reverseWords(s2);
    cout << "s2= "<< s2 << endl;
    cout << "ans= "<<ans << endl;

    string s3 = "1234";
    string s4 = "2";

    ans = multiply(s3,s4);
    cout << "ans= "<< ans << endl;

    string s5 = "we are happy";
    res = lengthOfLastWord(s5);
    cout << s5 << ",the length of last word is :"<< res << endl;

#if 0
    vector<string> strVec;
	string i;
	int a;
	while (cin >> i) 
    {
		strVec.push_back(i);
		cout << strVec[a] << endl;
		++a;
	}
#endif
	system("pause");

    return 0;
}







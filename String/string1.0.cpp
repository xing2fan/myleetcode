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
�ַ���

1.��Ҫ�ĸ��
����
�Ӵ���������
�����У���������
ǰ׺����Trie ����
��׺���ͺ�׺����
ƥ��
�ֵ���
2.��Ҫ�Ĳ�����
�������йصĲ�������ɾ�Ĳ�
�ַ��滻
�ַ�����ת

3.int �� long ���ܱ���������Χ���ޣ����Գ���ʹ���ַ���ʵ�ִ�����
���������صļӼ��˳���������Ҫģ�����Ĺ��̡�

4.������Ӵ�������������С�������Ӵ��������������

*/



//3.����һ���ַ����������ҳ����в������ظ��ַ��� ��Ӵ� �ĳ��ȡ�
int lengthOfLongestSubstring(string s) 
{

	vector<int> m(256,-1);//256��int������ģ���ϣ��,��¼�ַ����ֵ�λ��
	int left = -1;
	int res = 0;
	
	for(int i = 0;i < s.size();i++)
	{
		left = max(left,m[s[i]]);//���»���������߽�
		m[s[i]] = i;			 //�����ұ߽�
		res = max(res,i - left);//���»������ڴ�С
	}
	return res;
}


string longestPalindrome(string s) 
{
   int n = s.size();
   vector<vector<int>> dp(n, vector<int>(n));
   string ans;

   //ע�⣺��״̬ת�Ʒ����У������Ǵӳ��Ƚ϶̵��ַ����򳤶Ƚϳ����ַ�������ת�Ƶģ�
   //���һ��Ҫע�⶯̬�滮��ѭ��˳��
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

	//���յ�ǰ�ַ��Ƿ���Ȼ���Ϊ�������
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
   long res = 0;//Ҫʹ��long������int

   int flag = 1;//Ĭ��������
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
   //Ҫ��-+1������������������
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
		   return res;//�����ǰ�ַ��������֣��ͷ���֮ǰ��ֵ
	   }

	   //Ҫÿ�ζ��ж�resֵ�������˳�
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

   //�������⣬4�ǵ����ģ�9Ҳ�ǵ����ģ�1,4,5,9,10,40,50,90,100,400,500,900,1000,4000...
   //����ȷ���� 1 �� 3999 �ķ�Χ�ڹʶ�value���Ϊ1000�Ϳ���

   int values[] = {1000,  900,	500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
   string reps[] = {"M", "CM", "D",  "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};

   int len = sizeof(values)/sizeof(values[0]);

   string res;
   for (int i = 0; i < len; i++ )
   { 
	   while(num >= values[i])
	   {
		   res += reps[i];//����ҵ���������

		   num -= values[i];//����numֵ
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
	   	return "";//�������vectoΪ�գ��򷵻ء���
   	}
   	string res = strs[0];//ѡ���һ���ַ�����Ϊ���ձ�׼

   	for(int i = 1;i < strs.size();i++)//����ÿ���ַ���
   	{
	   for(int j = 0;j < res.length();j++)//��һ���ַ����ĳ���
	   {
		   if(res[j] == strs[i][j])//�����Աȵ�һ���ַ���ÿ���ַ�
		   {
			   continue;
		   }
		   else
		   {
			   res.erase(j);//�ҵ���һ�������ϵ��ַ�λ�ã���pos=j����ʼɾ��ֱ����β
			   break;//�������ַ�����������һ���ַ���
		   }
	   }
   }
   return res;
}



//����һ���ַ���������ұ�
vector<string> table = {"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};

//����1���������飬����2��������� ����3��ĳ��ѡ���ַ��� ����4��index
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

	//����ǰ����������ַ�����iλ�õ������ַ� ӳ��ɲ��ұ��е�3���ַ���ɵ��ַ�
	//digits[index] - '2'�ǵ�ǰ���ֵ�ֵ 
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
	//��һ�������巵�ؽ������,��ά�������Ի���Ҫһ��һά�м�ֵstring
	vector<string> res;

	//�ڶ���������
	traceback( digits,res, "", 0);

	//���������ؽ��
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
�����ַ���ֻ�������ź������������ַ����������ս���ض���������n����������n����
�������Ƕ�����������left��right�ֱ��ʾ:ʣ���������ŵĸ�����

1.�����ĳ�εݹ�ʱ��ʣ�������ŵĸ������������ŵĸ�����˵����ʱ���ɵ��ַ����������ŵĸ������������ŵ�
�������������'()('�����ķǷ����������������ֱ�ӷ��أ�����������

2.���left��right��Ϊ0����˵����ʱ���ɵ��ַ�������n�������ź�n�������ţ����ַ����Ϸ������������
�󷵻ء�

3.����������������������:
����ʱleft����0������õݹ麯����ע������ĸ��£�
����ʱright����0������õݹ麯����ͬ��Ҫ���²�����
*/
void generateParenthesisDFS(int left, int right, string out, vector<string> &res) 
{
   if (left > right)//�����Ÿ���������������
   {
	   return;
   }
   if (left == 0 && right == 0)//��ʱ��Ҫ����������
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
	//kmp�㷨
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
			j = 0;//��ͷ��ʼ����
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
	
	// �ȿ���ǰ�ַ����� ASCII ������Ƿ���� ����ȥ��
	for(int i = 0, j; i<=s.size()-lens; cur -= s[i], cur += s[lens + i++])
	{
		// ����ȥ��
		if(cur != target) 
        {
            continue;
        }
		// ȷ��һ�£��Ƿ�Ϊ���ƥ��
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
        //ֻ��Ϊ�����Ų�����չdp[i - 1]
        if (s[i] == ')') 
        {
            //��i���ַ������i?dp[i?1]?1���ַ�ƥ�䣬
            //��ôdp[i?1]������������չ������dp[i]=dp[i?1]+2
            if (i - dp[i - 1] - 1 >= 0 && s[i - dp[i - 1] - 1] == '(')
            {
                dp[i] = dp[i - 1] + 2;
            }

            //����ǰ��dp[i]��dp[i - dp[i]]���кϲ����һ����������Ч����
            if (i - dp[i] >= 0)
            {
                dp[i] += dp[i - dp[i]];
            }
        }
    }


    //�Ը����ַ�Ϊβ�����Ч���ų��ȣ��ó��ַ��������Ч���ų���
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

    //������m+nλ,��λ��ˣ�����Ӧ����һλ���������λ���ܴ��ڽ�λ�����Խ��Ӧ����m+nλ��
	vector<int> nums_res(max_len, 0);

    ////ģ����������һλ��ʼ����
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

    //��λ
	for (int i = max_len -1; i>0; i--) 
    {
		nums_res[i - 1] += nums_res[i] / 10;
		nums_res[i] %= 10;
	}


	int i = 0;
    //�������num_res��������m+n���±���0~m+n-1���洢�ķ�ʽ�ͼ���һ������ȡ�Ĵӵ�λ����λ��˳��
    /*���ڡ�0���͡�0������ļ��ж�*/ 
	while (i < max_len && nums_res[i] == 0)
    {
        i++;
        if(i == max_len)  
        {
            return "0";
        }
    }
        
            
	//ת�����ַ���
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
			++ i, ++ j;// ָ��ͬʱ��������1����ʾƥ��
		}
		else if (j < n && p[j] == '*') 
		{
			// ��¼���ݵ�λ��
			i_star = i;// ��¼�Ǻ�
			j_star = j++;// ����j�Ƶ���һλ,׼���¸�ѭ��s[i]��p[j]��ƥ��
						 //(Ҳ����ƥ��0���ַ�)
		}
		else if (i_star >= 0) 
		{
			// �����ַ���ƥ����û���Ǻų���,����istar>0 
			// ˵��*ƥ����ַ������ܳ��� ����
			i = ++ i_star;//i���ݵ�i_star+1����Ȼƥ���ַ������������ڶ�ƥ��һ�����������һ
			j = j_star + 1;//j���ݵ�j_star+1 ����ʹ��p��*��Ĳ��ֿ�ʼ����s��i_star
		} 
		else 
			return false;
	}
	while (j < n && p[j] == '*') ++ j;// ȥ�������Ǻ�

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
        else//��ǰ�ǿո�
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
        if (s[i] == ' ')//�ո�
        {
            //ǰһ�������֣�С���㣬ָ��e�������ţ��ҵ�ǰ�ǿո� �Һ��治�ǿո�
            if (i < n - 1 && s[i + 1] != ' ' && (num || dot || exp || sign)) 
            {
                return false;
            }
        } 
        else if (s[i] == '+' || s[i] == '-')//������
        {
            //��ǰ�ַ���������ʱ��ǰ��һ�����ֱ�����e���߿ո����
            if (i > 0 && s[i - 1] != 'e' && s[i - 1] != ' ') 
            {
                return false;
            }
            sign = true;
        } 
        else if (s[i] >= '0' && s[i] <= '9')//���� 
        {
            num = true;
            numAfterE = true;
        } 
        else if (s[i] == '.')//С����
        {
            //���ǰ����С���������ָ��e���У���Ϊe����ı�����������
            if (dot || exp) 
            {
                return false;
            }
            dot = true;
        } 
        else if (s[i] == 'e')//eָ�� 
        {
            //ǰ����ָ��e������û�����֣���Ϊeǰ�����������
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
    //path = "/a/./b/../c/", => "/a/c" ��  path = "/a/./b/c/", => "/a/b/c"
    // a . b .. c ==>/a/c              ��  a . b c ===>/a/b/c
    //�����.����ջ �����..��Ҫ��ջ���ַ��� 
    //����ǿյĻ�����"/"������ж��"/"ֻ����һ����
    //��ô���ǿ��԰�·����������һ������"/"�ָ���ڶ����ַ����������Ƿֱ���ȡ����һһ������
    

    vector<string> v;//����ַ���
    int i = 0;
    while (i < path.size()) 
    {
        //step1.����//�ָ��ַ���
        while (path[i] == '/' && i < path.size()) 
        {
            ++i;//����///
        }
        if (i == path.size()) 
        {
            break;
        }

        int start = i;//���� / �Ժ�
        while (path[i] != '/' && i < path.size()) 
        {
            ++i;
        }
        int end = i - 1;


        //step2.�����м�ֵ,//�����.����ջ �����..��Ҫ��ջ���ַ��� 
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

    //���ջ�ǿյģ�ֱ�ӷ���/
    if (v.empty()) 
    {
        return "/";
    }

    //ʹ�� / ��������ջ�е��ַ���
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

    // ��һ���ַ���Ϊ�մ�
    if (n * m == 0) 
    {
        return n + m;
    }

    // DP ����
    int D[n + 1][m + 1];

    // �߽�״̬��ʼ��
    for (int i = 0; i < n + 1; i++) 
    {
        D[i][0] = i;
    }
    for (int j = 0; j < m + 1; j++) 
    {
        D[0][j] = j;
    }

    // �������� DP ֵ
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

	//��ǰλΪֹ�Ľ��뷽������=��һ���Ľ��뷽������+��һ�������ĸ���
	//dp[i] = dp[i-1] + (dp[i-2] if ˫�ַ��ϸ� else 0)
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

//��֤���Ĵ�
//����һ���ַ�������֤���Ƿ��ǻ��Ĵ���ֻ������ĸ�������ַ������Ժ�����ĸ�Ĵ�Сд��
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
        //����дͳһת����Сд
        //a��ASCII����ֵ��97��A��ASCII����ֵ��65
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
        //��C++��stringת��ΪC���ַ�������
        //C�⺯�� int atoi(const char *str) �Ѳ��� str ��ָ����ַ���ת��Ϊһ������������Ϊ int               // �ͣ�������������string�മ������ʹ��string�����е�c_str()��������ת����
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

    //i������λ����ĸ�ĳ�ʼλ�ã�j�����ƶ�
    while(i < S.size())
    {
        while(S[i] == S[j])   
        {
            j++;//��ָ��
        }
        res += (S[i] + to_string(j - i));//j-i�ַ����ִ���

        i = j;//�ƶ���ָ��
    }

    //����ѹ��������ַ���û�б�̣��򷵻�ԭ�ȵ��ַ�����
    if(res.size() >= S.size())  
    {
        return S;
    }

    return res;
}


bool oneEditAway(string first, string second) {

	if(first == second)//�ַ������ֱ�ӷ���
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
		//i��������ɨ�裬�ҵ������ַ�����ȵ�����λ��
		++i;
	}

	//�����ַ����Ӻ���ǰɨ��
	while(j >= 0 && k >= 0 && first[j] == second[k])
	{ 
		//j��k��������ɨ��
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
	int cnt = 0;//cnt��¼�ַ�����
	//i��ֵ��n ��  s.size()+n
	for(int i = n; cnt < s.size(); i++)
	{
		ans += s[i % s.size()];
		cnt ++;
	}
	return ans;
}


string reverseWords(string s) {
    reverse(s.begin(), s.end());

    int idx = 0;  //��ʾ�ǿյ���ǰ�Ŀո�
    int i = 0, j = 0;  //�ǿյ��ʵķ�Χ

    for(i = 0; i < s.size(); ++ i)
    {
        //���"   hello world!"
        if(s[i] == ' ')
        {
            continue;  //�����ո��������
        }

        //�ǿ�ͷ
        if(idx != 0)
        {
            s[idx++] = ' '; //���ʿ�ͷ���ո�
        }

        j = i;  //�ǿյ��ʵĿ�ͷ
        while(j < s.size() && s[j] != ' ')
        {
            s[idx++] = s[j++];//���ǿյ�����ǰ�ƣ�����ǰ��Ŀո�
        }

        reverse(s.begin() + idx - (j - i), s.begin() + idx);

        i = j;  //����i��λ��
    }

    
    s.erase(s.begin() + idx, s.end());  //ɾ��ĩβ�Ŀո�
    return s;
}



//leftΪ����������rightΪ��������
void dfs(vector<string> &ans,int n,int left, int right, string now)
{
	if(left == n && left == right)
	{
		ans.push_back(now);
		return;
	}

	if(left < n)
	{
		dfs(ans, n, left + 1, right,now + "("); //������
	}
	if(right < n && right < left) 
	{
		dfs(ans, n, left, right + 1,now + ")");//��������
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

	//step1.ͳ�ƿո�ĸ���
	int cnt = 0;
	for (int i = 0; i < length; ++i) 
	{
		if (S[i] == ' ')
		{
			++cnt;
		}
	}

	//step2.�������ַ����ĳ���
	int newLen = length + cnt * 2;

	//step3.�Ӻ���ǰ����
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
	//��ͷ�����������ո���� %20
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







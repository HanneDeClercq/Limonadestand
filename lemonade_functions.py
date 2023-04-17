from scipy.stats import bernoulli as bernoulli
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import random

global p_W_obs, p_S_W_obs, p_B_W_obs, p_C_WB_obs

global noise_var

noise_var=1.9
p_W_obs = np.array([0.55,0.45])

p_S_W_obs = np.array([[1,0],[0,1]]).transpose()

p_B_W_obs = np.array([[0.5,0.5],[1,0]]).transpose()

p_C_WB_obs = np.zeros([2,2,2])

p_C_WB_obs[0][0][:] = [1,0]
p_C_WB_obs[0][1][:] = [0.5,0.5]
p_C_WB_obs[1][0][:] = [0,1]
p_C_WB_obs[1][1][:] = [0,1]

p_C_WB_obs = p_C_WB_obs.transpose(2,0,1)

def sample_cf_policy(T,p_W,p_S_W,p_B_W,p_C_WB,my_dict):
    # S=1      => do nothing
    # S=0, B=0 => do(B=1)
    # S=0, B=1 => do nothing
    Weather = np.random.choice(2,T,p=p_W)
    Sunglasses = np.zeros((T,))
    Beanie = np.zeros((T,))
    Craving = np.zeros((T,))
    i=0
    for w in Weather:
        Sunglasses[i]=np.random.choice(2,1,p=p_S_W[:,w])
        b=np.random.choice(2,1,p=p_B_W[:,w])
        Beanie[i]=b
        Craving[i]=np.random.choice(2,1,p=p_C_WB[:,w,b].flatten())
        i+=1    
    
    promotion_locations=[False]*T
    for i in range(T):
        if Sunglasses[i]==0 and Beanie[i]==0 and my_dict['S0','B0']:
            w = Weather[i]
            Craving[i]=np.random.choice(2,1,p=p_C_WB[:,w,1].flatten())
            promotion_locations[i]=True
        elif Sunglasses[i]==0 and Beanie[i]==1 and my_dict['S0','B1']:
            w = Weather[i]
            Craving[i]=np.random.choice(2,1,p=p_C_WB[:,w,1].flatten())
            promotion_locations[i]=True
        elif Sunglasses[i]==1 and Beanie[i]==0 and my_dict['S1','B0']:
            w = Weather[i]
            Craving[i]=np.random.choice(2,1,p=p_C_WB[:,w,1].flatten())
            promotion_locations[i]=True
        elif Sunglasses[i]==1 and Beanie[i]==1 and my_dict['S1','B1']:
            w = Weather[i]
            Craving[i]=np.random.choice(2,1,p=p_C_WB[:,w,1].flatten())
            promotion_locations[i]=True
        else:
            pass
    Price = np.array([10+10*Craving[i] + noise_var*np.random.randn() for i in range(T)])
    
    return Weather,Sunglasses,Beanie,Craving,Price,promotion_locations

def sample_data(T,p_W,p_S_W,p_B_W,p_C_WB):
    Weather = np.random.choice(2,T,p=p_W)
    Sunglasses = np.zeros((T,))
    Beanie = np.zeros((T,))
    Craving = np.zeros((T,))
    i=0
    for w in Weather:
        Sunglasses[i]=np.random.choice(2,1,p=p_S_W[:,w])
        b=np.random.choice(2,1,p=p_B_W[:,w])
        Beanie[i]=b
        Craving[i]=np.random.choice(2,1,p=p_C_WB[:,w,b].flatten())
        i+=1

    Price = np.array([10+10*Craving[i] + noise_var*np.random.randn() for i in range(T)])
    return Weather,Sunglasses,Beanie,Craving,Price

def obs_data(T):
    p_W = p_W_obs
    p_S_W = p_S_W_obs
    p_B_W = p_B_W_obs

    p_C_WB = p_C_WB_obs

    return sample_data(T,p_W,p_S_W,p_B_W,p_C_WB)

def policy_data(T,my_dict):
    p_W = p_W_obs
    p_S_W = p_S_W_obs
    p_B_W = p_B_W_obs

    p_C_WB = p_C_WB_obs

    return sample_cf_policy(T,p_W,p_S_W,p_B_W,p_C_WB,my_dict)

def int_on_beanie(T):
    p_W = p_W_obs
    p_S_W = p_S_W_obs
    p_B_W = p_B_W_obs

    p_C_WB = p_C_WB_obs
    
    p_B_W = np.array([[0,1],[0,1]]).transpose()
    Weather,Sunglasses,Beanie,Craving,Price = sample_data(T,p_W,p_S_W,p_B_W,p_C_WB)
    return Weather,Sunglasses,Beanie,Craving,Price

def int_on_sunglasses(T):
    p_W = p_W_obs
    p_S_W = p_S_W_obs
    p_B_W = p_B_W_obs

    p_C_WB = p_C_WB_obs
    
    p_S_W = np.array([[0,1],[0,1]]).transpose()
    Weather,Sunglasses,Beanie,Craving,Price = sample_data(T,p_W,p_S_W,p_B_W,p_C_WB)
    return Weather,Sunglasses,Beanie,Craving,Price

def visualize_obs_data(Weather,Sunglasses,Beanie,Craving,Price):
    T = len(Price)
    plt.figure()

    pwb = [Price[i] if Beanie[i]==1 else 0 for i in range(T)]
    pwoutb = [Price[i] if Beanie[i]==0 else 0 for i in range(T)]

    pwb=np.cumsum(pwb)
    pwoutb=np.cumsum(pwoutb)
    for i in np.arange(T-1,T):
        plt.plot(range(i),pwb[0:i])
        plt.plot(range(i),pwoutb[0:i])
        plt.xlabel('No. of Customers')
        plt.ylabel('Total price paid')        
        plt.legend(['Cust. w/ Beanie','Cust. w/out Beanie'])
        if i==T-1:
            plt.show()
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.2)
            plt.clf()
        #plt.close()

    price_w_beanie=Price[np.where(Beanie==1)]
    price_wout_beanie=Price[np.where(Beanie==0)]

    print('Avg. price paid by customers w/ Beanie: %2.4f'%np.mean(price_w_beanie))
    print('Avg. price paid by customers w/out Beanie: %2.4f'%np.mean(price_wout_beanie))

    plt.figure()

    pws = [Price[i] if Sunglasses[i]==1 else 0 for i in range(T)]
    pwouts = [Price[i] if Sunglasses[i]==0 else 0 for i in range(T)]

    pws=np.cumsum(pws)
    pwouts=np.cumsum(pwouts)
    for i in np.arange(T-1,T):
        plt.plot(range(i),pws[0:i])
        plt.plot(range(i),pwouts[0:i])
        plt.xlabel('No. of Customers')
        plt.ylabel('Total price paid')
        plt.legend(['Cust. w/ Sunglasses','Cust. w/out Sunglasses'])
        if i==T-1:
            plt.show()
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.2)
            plt.clf()
        #plt.close()

    price_w_sunglasses=Price[np.where(Sunglasses==1)]
    price_wout_sunglasses=Price[np.where(Sunglasses==0)]

    print('Avg. price paid by customers w/ Sunglasses: %2.4f'%np.mean(price_w_sunglasses))
    print('Avg. price paid by customers w/out Sunglasses: %2.4f'%np.mean(price_wout_sunglasses))
    
    print('\n Avg. price paid by customers overall: %2.4f'%np.mean(Price))

def visualize_int_data(Weather,Sunglasses,Beanie,Craving,Price,promotion):
    p = np.cumsum(Price)
    for i in np.arange(T-1,T):
        plt.plot(range(i),p[0:i])
        #plt.plot(range(i),pwouts[0:i])
        plt.legend(['Price w/ %s Promotion'%promotion])
        if i==T-1:
            plt.show()
            plt.close()
        else:
            plt.show(block=False)
            plt.pause(0.2)
            plt.clf()
    print('Avg. price paid by customers after %s Giveaway: %2.4f'%(promotion,np.mean(Price)))

def visualize_profit_cf_policy(Price, Sunglasses, Beanie,promotion_locations):
    # beanie promotion
    promotion='Beanie'
    T = len(Price)
    baseline_price = 12
    promotion_cost = 4.7
    profit=[]
    for i in range(T):
        if promotion_locations[i]==True:
            profit.append(Price[i]-baseline_price-promotion_cost)
        else:
            profit.append(Price[i]-baseline_price)
    p = np.cumsum(profit)
    for i in np.arange(1,T):
        if np.sum(p[i])<-100:
            print('You went bankrupt!!')
            img = mpimg.imread('bankrupt.png')
            plt.axis('off')
            imgplot = plt.imshow(img)
            plt.pause(2)
            plt.close()
            break
        plt.plot(range(i),p[0:i])
        plt.xlabel('No. of Customers')
        plt.ylabel('Profit')        
        #plt.plot(range(i),pwouts[0:i])
        plt.legend(['Profit w/ %s Promotion'%promotion])
        plt.show(block=False)
        plt.pause(0.2)
        plt.clf()
        if i==T:
            plt.pause(2)
            plt.close()
    if p[i]>0:
        print('You earned %4.2f$!!'%p[i])
    else:
        print('You lost %4.2f$.'%np.abs(p[i]))
    
def visualize_profit(Price,promotion):
    T = len(Price)
    baseline_price = 12
    if promotion == 'Sunglasses':
        promotion_cost = 4.3
    elif promotion == 'Beanie':
        promotion_cost = 4.3
    profit = [i-baseline_price - promotion_cost for i in Price]
    p = np.cumsum(profit)
    for i in np.arange(1,T):
        if np.sum(p[i])<-100:
            print('You went bankrupt!!')
            img = mpimg.imread('bankrupt.png')
            plt.axis('off')
            imgplot = plt.imshow(img)
            plt.pause(2)
            plt.close()
            break
        plt.plot(range(i),p[0:i])
        plt.xlabel('No. of Customers')
        plt.ylabel('Profit')        
        #plt.plot(range(i),pwouts[0:i])
        plt.legend(['Profit w/ %s Promotion'%promotion])
        plt.show(block=False)
        plt.pause(0.2)
        plt.clf()
        if i==T:
            plt.pause(2)
            plt.close()
    if p[i]>0: #np.sum(p[0:i])>0:
        print('You earned %4.2f$!!'%p[i])
    else:
        print('You lost %4.2f$.'%np.abs(p[i]))

def slow_type(t):
    typing_speed = 200 #wpm
    for l in t:
        sys.stdout.write(l)
        sys.stdout.flush()
        time.sleep(random.random()*10.0/typing_speed)
    print('')
    
def narration():
    slow_type('Congratulations!')
    time.sleep(0.5)
    slow_type('You just purchased a lemonade stand.')
    time.sleep(0.5)
    slow_type('The previous owner gave you some data about which customers paid how much money in the past.')
    time.sleep(0.5)
    slow_type('You can look at this data and decide which promotional items to give out to increase your profit.')
    time.sleep(0.5)
    slow_type('Please observe this data. When you are ready, close the pop-up window to proceed.')

def narration2():
    slow_type('Equipped with causal knowledge, you decide to come up with a policy.')
    time.sleep(0.5)
    slow_type('We would like to give away beanie only for some customers.')
    time.sleep(0.5)
    slow_type('Answer Y/N on whether to give away Beanie to customers with the following features:')
    
def main():
    #observational data
    narration()
    T = 100

    Weather,Sunglasses,Beanie,Craving,Price = obs_data(T)

    visualize_obs_data(Weather,Sunglasses,Beanie,Craving,Price)

    main_flag=1
    while main_flag:    
        print('Enter which promotional item to give away S for sunglasses, B for Beanie:')
        flag=1
        while flag:
            val = input()
            if val == 'S' or val == 's':
                promotion = 'Sunglasses'
                flag=0
            elif val == 'B' or val == 'b':
                promotion = 'Beanie'
                flag=0
            else:
                print('Invalid input, please try again:')
        print('You decided to give away %s. Let us observe the profit curve: '%promotion)
        if promotion == 'Sunglasses':
            Weather,Sunglasses,Beanie,Craving,Price = int_on_sunglasses(T)
            visualize_profit(Price,'Sunglasses')
        elif promotion == 'Beanie':
            Weather,Sunglasses,Beanie,Craving,Price = int_on_beanie(T)
            visualize_profit(Price,'Beanie')
        local_flag=1
        while local_flag:
            val=input('Would you like to try again? (Y/N)?\n')
            if val=='y' or val=='Y':
                local_flag=0
                break
            elif val=='n' or val=='N':
                local_flag=0
                main_flag=0
                print('Thanks for playing!')
                break
            else:
                print('Invalid entry, please try again.')

def main2():
    narration2()
    T = 100
    main_flag=1
    while main_flag:
        my_dict={}
        for i in range(2):
            for j in range(2):
                my_key='S'+str(i),'B'+str(j)
                my_dict[(my_key)]=False

#         my_dict[('S1','B1')]=False
#         my_dict[('S1','B0')]=False
#         my_dict[('S0','B1')]=False        
#         my_dict[('S0','B0')]=False
        for i in range(2):
            for j in range(2):
                my_key='S'+str(i),'B'+str(j)
                print('Customers w/ S=%s, B=%s:'%(str(i),str(j)))
                flag=1    
                while flag:
                    val = input()
                    if val == 'Y' or val == 'y':
                        my_dict[my_key]=True
                        flag=0
                    elif val == 'N' or val == 'n':
                        pass
                        flag=0
                    else:
                        print('Invalid input, please try again:')

#         print('Customers w/ S=1, B=0:')
#         flag=1    
#         while flag:
#             val = input()
#             if val == 'Y' or val == 'y':
#                 my_dict[('S1','B0')]=True
#                 flag=0
#             elif val == 'N' or val == 'n':
#                 pass
#                 flag=0
#             else:
#                 print('Invalid input, please try again:')
#         print('Customers w/ S=0, B=1:')
#         flag=1    
#         while flag:
#             val = input()
#             if val == 'Y' or val == 'y':
#                 my_dict[('S0','B1')]=True
#                 flag=0
#             elif val == 'N' or val == 'n':
#                 pass
#                 flag=0
#             else:
#                 print('Invalid input, please try again:')
#         print('Customers w/ S=0, B=0:')
#         flag=1    
#         while flag:
#             val = input()
#             if val == 'Y' or val == 'y':
#                 my_dict[('S0','B0')]=True
#                 flag=0
#             elif val == 'N' or val == 'n':
#                 pass
#                 flag=0
#             else:
#                 print('Invalid input, please try again:')
        s=''
        for i in range(2):
            for j in range(2):
                my_key='S'+str(i),'B'+str(j)
                if my_dict[(my_key)]==True:
                    s+=' '
                    s+='S'+str(i)+'B'+str(j)
                    
        print('You decided to give away beanie for customers: %s'%s)
        
        Weather,Sunglasses,Beanie,Craving,Price,promotion_locations=policy_data(T,my_dict)
        visualize_profit_cf_policy(Price, Sunglasses, Beanie,promotion_locations)
        local_flag=1
        while local_flag:
            val=input('Would you like to try again? (Y/N)?\n')
            if val=='y' or val=='Y':
                local_flag=0
                break
            elif val=='n' or val=='N':
                local_flag=0
                main_flag=0
                print('Thanks for playing!')
                break
            else:
                print('Invalid entry, please try again.')            

            
# if __name__ == "__main__":
#     #main()
#     main2()
    
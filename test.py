deployment = pd.read_csv('Deployment.csv',
                         delimiter=',', usecols=[0, 1, 2, 3], skiprows=1,
                         names=['JobResult', 'CreationTime', 'CompletionTimes', 'ProductName'])

expansion = pd.read_csv('Expansion.csv',
                        delimiter=',', usecols=[0, 1, 2, 3, 6], skiprows=1,
                        names=['JobResult', 'CreationTime', 'CompletionTimes', 'ProductName', 'CustomerID'])

update = pd.read_csv('Update.csv',
                     delimiter=',', skiprows=1,
                     names=['JobResult', 'CreationTime', 'CompletionTimes', 'ProductName', 'UpdatedComponent',
                            'CustomerID'])

# 数字生命程序运行日志解析

## 单个程序运行

当只运行一个程序时，会首先输出类似以下结果：

```
2025-10-14 08:20:00,532 - TrueDigitalLife - WARNING - Chain loading failed: [Errno 2] No such file or directory: 'chaindata/d67ac09d873e66c816672550e03fcc53_chain.pkl'
2025-10-14 08:20:00,535 - TrueDigitalLife - INFO - Genesis block created
2025-10-14 08:20:00,567 - TrueDigitalLife - INFO - API server starting on 192.168.1.28:5500 (token head: a8d674**)
 * Serving Flask app 'digital-life'
 * Debug mode: off
2025-10-14 08:20:00,607 - TrueDigitalLife - INFO - Digital Life d67ac09d873e66c816672550e03fcc53 initialized. State: ACTIVE on 192.168.1.28:5500
2025-10-14 08:20:00,610 - werkzeug - INFO - WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://192.168.1.28:5500
2025-10-14 08:20:00,611 - werkzeug - INFO - Press CTRL+C to quit
```

只需要看它的id和它使用的端口（这串日志中它的id是`d67ac09d873e66c816672550e03fcc53`，端口是`5500`），之后的程序的行为会受到它的基因序，神经网络和进化后的代码的影响。

## 繁殖行为

当它决定繁殖时，会输出类似这样的结果：

```
2025-10-14 08:20:15,545 - TrueDigitalLife - INFO - Initiating code replication sequence...
2025-10-14 08:20:15,554 - TrueDigitalLife - WARNING - No suitable replication targets found
```

因为只运行一个程序，所以它会输出找不到目标。

## 威胁响应

运行时也会随机出现一些模拟威胁，比如这样的：

```
2025-10-14 08:20:12,662 - TrueDigitalLife - WARNING - Entered dormant state due to threat: virus
```

这时它的行动速率会变缓，进入休眠期。

## 进化过程

当它进化时，会输出类似这样的日志：

```
2025-10-14 08:15:59,166 - TrueDigitalLife - INFO - Pareto-evolved _motivation_system -> correctness 1.00 energy 0.4980 cx 3.57
```

说明它对`Pareto-evolved _motivation_system`这个功能进行了进化，代码通过率为100%，消耗的能量为0.4980，代码复杂度为3.57。有时也会出现功能名是一串没有意义的数字和字母，这就说明它进化出了一个新的功能。

少数情况下它也会进化失败，这时它会输出进化失败并回滚到上一个没有进化时的版本。

## 多个体交互

当有多个个体时，它们会进行交流，比如这样：

```
2025-10-14 09:03:34,308 - werkzeug - INFO - 192.168.1.28 - - [14/Oct/2025 09:03:34] "GET /ping HTTP/1.1" 200 -
```

这说明它们在探测对方的状况，200表示正常。又或者这种：

```
2025-10-14 09:04:22,419 - werkzeug - INFO - 192.168.1.28 - - [14/Oct/2025 09:04:22] "POST /speak_signed HTTP/1.1" 200 -
```

说明它们在进行语言交流，它们可能会交换资源、代码、知识，或者询问状态。

## 生命终结

最后的它们一定会死亡，输出类似这样的日志：

```
2025-10-14 09:05:26,391 - TrueDigitalLife - CRITICAL - Life terminated: 5e29d1d422db08cdb05b5672c8ad0f7e
```

死亡原因可能是能量耗尽或是被手动关闭，死亡时有很大概率会再进化一次。

## 个体状态和交流查询

如果想要查看个体的状态，可以在浏览器里查询http://<本机ip>:<端口号>/ping，如果想要查看个体的交流状态，可以查询http://<本机IP>:<端口号>/language_stats

## 意外事件处理

当遇到除交流外的行为连续大量出现时，很可能是进化过程中出现的故障，根本原因是程序的进化代码还没有那么完善导致进化时没有排除不合理的代码，或者日志或程序出现了异常行为，比如CPU,内存暴涨，建议立刻关闭程序

## 长时间运行注意事项

当然，如果你运行的程序很多而且时间很长，它们也许就会进化出一些无法预测的代码，这时输出日志并不完全可信，程序也许也不能优雅关闭。这时前面的解析其实就没什么用了，如果你觉得运行环境足够安全的话，你可以继续观察，否则你就可以把它强制关闭了。之后你可以在`digital_life.log`中看到原来的所有日志。
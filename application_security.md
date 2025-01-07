# Security by Design

## Secure SDLC
The Software Development Lifecycle (or SDLC) is a framework that specifies the steps involved in software development at each stage. SDLC is a well-structured sequence of stages that leads to the rapid development of high-quality software that has been completely tested and is ready for production. Different stages of the SDLC are: 
- Requirements, where the project team begins to comprehend the customer's expectations for the project
- Design, where decisions are based on the requirements list that you created during the Requirements stage
- Develop, where a design is put into action 
- Test, where developers test their code and programming to see whether it meets client needs or if the functionality is smooth
- Deploy, which you execute when the product is ready to go live after it has been tested by the project team and has passed each testing phase

    <p align="center">
    <img src="./assets/application_security/SDLC-sec.png" alt="drawing" width="600" height="300" style="center" />
    </p>

How can you map DevOps into the phases of a secure SDLC? 
- During the Requirements phase, 
    - You perform risk assessment and consider how people might attack the code. Make sure you've determined the security needs and standards, as well as the type of information you're dealing with
    - Define the security requirements where you identify the information to protect. And perform attack profiling to determine what might be going on throughout the design threat modeling process 

- The design stage is about designing in a secure manner:

    - Perform threat modelling to secure your design architecture. During design threat modeling, ask, what are some of the elements that could make your architecture vulnerable? How can you securely design for the precautions that can be taken during this stage
    - Secure the deployment pipeline by ensuring that you have a secure design, that you've automated all the tests correctly, and that your CI/CD pipeline is searching for vulnerabilities
    - Create unit tests to counter common threats. With DevOps, security team members can instruct Dev team members about common threat types and help them create unit tests to counter them. 

- In the develop stage, you can 

    - Perform static analysis with tools that will check for security vulnerabilities in your code, look at it, and proclaim it insecure. Incorporate the validation of automated data to guarantee that the information in the system is both correct and useful. 
    - Use security tasks and a security in scrum. Scrum is a scrum framework variation that emphasizes secure software development throughout the SDLC. 

- In the test stage, you want to 
    - Incorporate vulnerability scans. You undertake security testing on your code, and you conduct a risk assessment before you launch it
    - Strive for failure: If you can break your application, attackers are likely to be able to do so as well. 
    - Parallelize security testing: To save time, run tests in parallel to shorten the test window by using code scanners alongside unit tests and functional verification tests (or FVTs). 

- When you get to the deploy stage, and your code is in production, you will ensure that it is functioning safely in various ways. You can 
    - Use automated launch of deployment scripts.
    - Use deploy and roll back, which means that for a file upload deployment, rollback will essentially revert the changes. So, if a file was previously uploaded, it will be erased; if a modification was made, it will be undone; and if a file was removed, it will be placed back
    - Perform production security tests, which imitate real-world hacking methods and approaches to reveal hidden flaws in your device or application. These tests can give you genuine insights and practical outcomes. 

## What is DevSecOps?
DevSecOps is all about DevOps with an emphasis on security. DevSecOps is a set of practices that automates security integration across the software development lifecycle (or SDLC) from original design to integration, testing, deployment, and software delivery. It's a discipline of adding security early in the application development lifecycle to reduce risks and integrate security closer to information technology (or IT) and the objectives of the enterprise. 

- Development means new software releases and software updates
-  Security indicates accessibility, integrity, and confidentiality
- Operation is performance scaling based on reliability


<p align="center">
<img src="./assets/application_security/benefits-devsecops.png" alt="drawing" width="600" height="300" style="center" />
</p>

Here are five of its most significant benefits: 
1. Delivering high-quality software quickly and at an affordable price. Resolving coding and security flaws may be time-consuming and costly. DevSecOps helps in reducing time spent on fixing security vulnerabilities by reducing the need to repeat a procedure to minimize development costs. You get increased delivery rates while reducing expenses
2. Increased security through proactive measures. DevSecOps integrates cybersecurity practices into the development lifecycle from the start by checking security, monitoring, deployment, and notification systems. As soon as concerns are detected, they are remediated 
3. Vulnerability patching at an accelerated pace. DevSecOps is the speed with which it manages newly discovered security flaws. Combining vulnerability screening and patching into the release cycle reduces the possibility of a threat actor to exploit vulnerabilities in public-facing production systems. DevSecOps strengthens accessibility and transparency from the beginning of development.

4. A modern approach to automation. If your enterprise employs a continuous integration/continuous delivery (or CI/CD) pipeline to deploy its software, you should 
    - Include cybersecurity testing in an automated test suite for operations teams. The automation of security procedures is heavily influenced by the project and organizational goals 
    - Automated testing checks that all included software dependencies are up to date and ensures that software passes security unit testing
    - And static and dynamic analysis tests and secures code before releasing it to production 
    - Plus, enabling Immutable infrastructure involves security automation to enhance complete security 

5. The cycle of recurrence and the ability to adapt. Well as enterprises develop, so do their security postures.
    - DevSecOps enables repeatable and adaptive processes, which guarantee that security is implemented uniformly throughout the environment as it develops and adapts to new requirements  
    - A mature DevSecOps system will feature strong automation, configuration management, orchestration, containers, immutable infrastructure, and even serverless compute environments, which give you speedier recovery after a security incident

## Understanding the role of Network Security

The Internet is a complicated network of networks that stretches around the globe via interconnected cables. All kinds of data traverse the Internet, such as emails, phone calls, streaming events, etc. The pioneers of the Internet had to create a system to allow for present and future types of communications to be used by everyone globally. They devised the Open Systems Interconnection, or the OSI model to solve that. The OSI model consists of seven layers to describe the process of sending and receiving data:

- Physical: it's puepose is to transmit bits of raw information

- Data Link: it transforms the transmitted raw data using data frames. The data frames are sequentially transmitted in groups of 100 or 1000 bytes. After the data frames are received, the receiver sends back an acknowledgment frame to confirm a correct reception of the data

- Network: handles data transmission and the control of the subnet. The network layer determines how many packets are routed to a destination from a source (route tables)

- Transport: accepts transmissions or data from the upper layers (application, presentation, session layers) and then chops them into smaller units or packets for passing to the network layer. Transport layer assures that all the units arrive correctly from end-to-end. This layer provides an error-free point-to-point channel that delivers data in the same order as they were sent. At connection time, the transport layer will choose the type of service

- Session: establishes multiple sessions from different machines while establishing consistent sessions if a crash occurs. This layer delivers benefits such as dialog control or transmission turn tracking and token management, which eliminates two users simultaneously attempting the same important operation. The session layer also provides synchronization for reestablishing sessions from long transmissions and allowing them to resume from the last point

- Presentation: focuses on the syntax and semantics of data being transmitted from one point to another. The serialization and deserialization process is performed on the data stream to rebuild the original format at the final destination. For instance, formats or file types such as jpeg, gif and ASCII text are widely and frequently used in the presentation layer. In addition, this layer also provides data compression encryption and decryption

 - Application: is the top layer of the OSI model. Mainly, developers use this layer for building and deploying applications. Browsers, web pages and other web-based applications transmit data on the application layer. Besides the web, file transfer, electronic mail or email, and network news are other applications that use Layer 7

 When a client requests a resource from a server, a connection is made between the client and the server, which occurs on the session layer of the OSI model. You can incorporate secure socket encryption at the presentation layer to keep user submitted data safe from potential man in the middle attacks. Another method for securing applications by developers is using port 443 and the secured version of HTTP, known as HTTPs. In eliminating unsecured communications on the application level, you as a software developer can build the necessary trust with your application users. The OSI model is an effective tool for understanding and discussing network communication concepts and is the basis for many modern network standards and technologies.


## Securing Layers for Application Development
- The first layer to secure for application developers is the web application layer. The web application layer could be composed of a front-end layer consisting of JavaScript, CSS, and HTML code running in web browsers using Hypertext Transport Protocol Secure or HTTPS across devices. 

- The backend layer of the web application layer typically consists of databases in the cloud that provide data to the front end during users' experiences with the application. 

- The middle layer of the web application provides a connection between the front end and the back end by using an application programming interface or API developed in languages such as Python, Java, or Ruby. As an application developer, you must test all the layers of a web application. 

<p align="center">
<img src="./assets/application_security/layers-security-applicatins.png" alt="drawing" width="600" height="300" style="center" />
</p>

Secure the first layer by running vulnerability scanners, tests, and allow other team developers to audit the web applications before deploying to production.

Safeguard the backend layer by securing the Cloud infrastructure. Be sure you protect administrator credentials when developing applications to connect to Cloud-based databases. Create security groups or Network Access Control List that restrict access to certain Cloud resources. 

- In your code, implement _two-factor authentication_ for all users of web applications. Be it phone or text authentication, you should also include strong authentication. Two-factor authentication is also important to authenticate third parties, such as GitHub and respective Cloud providers. 

- Secure the communications between clients and servers using a secure shell or SSH, HTTPS, Secure Sockets Layer, and Transport Layer Security (SSL/TLS). Data transferred over secure connections with SSL and TLS, guarantee that hackers attempting man-in-the-middle attacks do not intercept communications.  

- If Cloud sources are implemented for developing applications, Identification and Access Management, or IAM, should be configured for securing Cloud assets according to the needs and roles when developing. IAM roles are an important security mechanism to grant permissions to applications and systems within Cloud infrastructures. Finally, secret passwords, admin credentials, certificates, and encryption keys should be stored in secret storage services such as HashiCorp Vault or AWS secret manager

- Logging is considered another security layer, analyzing and storing for future inspection by application developers. It's also important to remember that every application should have a logging system to collect log messages for identifying any anomalies. Anomalies are unexpected events occurring within an application or system such as an attempt to log in as an administrator of a system without the necessary credentials. Lastly, access to the log messages should not be provided to all system users, but only those who can be trusted and need access for reviewing and analyzing. 

- The final layer of defense is intrusion detection. Intrusion detection is the ongoing detection of any cyberattacks, threats, and intrusions that compromise an application or a system. The three methods of intrusion detection are _endpoint security_, _network security_, and _system-call auditing_:
    - Endpoint security protect systems, servers, and various types of devices connected to a network
    - Network security is monitoring a network using a network tool such as _Nmap_ and _Snort_. 
    - System call auditing is the retrieval and review of system call information from a kernel such as the Linux kernel. 
    
## Security Pattern
A security pattern is essentially a set of rules that represent and define a reusable solution to recurring security threats or issues. As security design artifacts, a security pattern typically includes documenting security threats or issues. 

 Software developers must use security patterns to make their software easily identifiable, reusable, and interoperable with other applications and services. It's worth noting that security patterns simplify the complexities of managing security threats and attacks. 

Security patterns also offer actionable solutions and recommendations for implementing security controls, monitoring mechanisms, authentication processes, encryption protocols, and more. The primary goal of security patterns is to reduce or eliminate future security threats. These patterns directly relate to a specific threat rather than a vulnerability. Because security patterns take their base from previous incidents and vulnerabilities, organizations must develop new security patterns when new threats occur.

For example, security patterns can be classified under authentication, access control, or filtering network traffic within a network. Authorization, role-based access control, firewalls, and web services security such as SAML, XACML, and XML firewalls are some other examples of security patterns.

<p align="center">
<img src="./assets/application_security/security-patterns.png" alt="drawing" width="600" height="300" style="center" />
</p>

Security patterns provide a comprehensive framework for addressing unique security challenges, safeguarding customer information, and ensuring the organization's ecosystem integrity.

## TLS/SSL

Both are protocols for establishing secure connections between network computers, specifically a server and a client. When we say secure, we mean that If someone were to intercept the communications, it would be useless to them because it would be unreadable due to encryption.

In fact, TLS is a successor to SSL, the first version of TLS, TLS 1.0 was introduced in 1999. Today, when people refer to SSL or TLS/SSL, they are usually referring to modern TLS. So how does modern TLS work? At a high level, it uses four steps; you can follow these steps to ensure TLS stays secure in the software development lifecycle, or SDLC.

<p align="center">
<img src="./assets/application_security/how-tls-works.png" alt="drawing" width="600" height="300" style="center" />
</p>

So how do you ensure TLS remains secure in your application's SDLC? Basically, with two components: 
- First, you use Continuous Integration and Continuous Delivery, or CI/CD, to renew TLS certificates before their expiration date. They usually expire about every one or two years, however, it's a good practice to renew earlier. 
- Second, you need to make sure that your application keeps its TLS version support up to date, this means it should support the newest available version of TLS. Also, it should prefer the most robust ciphers and avoid vulnerable ciphers at all costs. This often means dropping support for outdated versions of TLS such as 1.0 and 1.1. TLS and SSL contribute to secure, trustworthy and seamless communication between client and server. By implementing these protocols, you can protect as well as ensure the confidentiality and integrity of your data.


## OpenSSL

By implementing cryptography, you can eliminate the potential for snooping or eavesdropping by bad actors. In addition, other network attacks such as Spoofing and Hijacking, can be stopped. The cryptography service is a confidentiality service that keeps data secret. Its purpose is to keep data secure from others without the necessary credentials, even when data traverses a nonsecure network. Cryptographic keys, specifically private keys, are the tools you can use to keep data confidential from cyberattacks. Without public and private keys, data remains encrypted for e-commerce transactions, ensuring users' information remains confidential. 

The next cryptographic service is integrity. Integrity guarantees that the data has not been modified or tampered with during and after the reception. For example, a file checksum is one method to verify that the downloaded file is the same as the one available online.

OpenSSL is a library of software that implements the Secure Socket Layer, or SSL protocol. It's an open-source toolkit to ensure secure communications with cryptography for all types of communications, from personal to commercial and e-commerce transactions. The library of software contains symmetric and public key cryptography, message digests, and hash algorithms. OpenSSL also includes a pseudorandom number generator and manages key material and common certificate formats.

You can also use OpenSSL directly from the command line after installing it on a local computer. In addition, you can run the OpenSSL command directly by running OpenSSL on Linux or Mac or OpenSSL.exe on Windows. The command line tool offers numerous options, and a configuration file is available to set configurations as defaults to better tailor to your specific requirements.

- Message digest algorithms are cryptographic hash functions used to compute checksums of data blocks. Besides computing hashes, you can also use message digest to sign and verify signatures. When you verify a signature, it's simply running the reverse of the command when signing. It's important to note that OpenSSL also supports symmetric ciphers that allow encrypted messages to use the same key.

- Next public key cryptography is a public cryptographic algorithm that uses public and private keys. Rivest, Shamir and Adleman, or RSA, is the most popular implementation of public key cryptography. RSA provides secrecy, authentication, and encryption for anyone to use. You can also use RSA to implement prime number generation to generate private keys using different sizes of key lengths depending on the level of encryption needed.

## Vulnerability Scanning 
Vulnerability scanning is the search for security vulnerabilities from within the code and from the outside of an application. Vulnerability scanners search in a variety of code languages such as C or C++, Java, Python, and PHP. Some common code vulnerabilities to scan for include structured query language (or SQL) injection, cross-site scripting and path traversal of files and directories in web applications.

- Vulnerability scans on the specific platform configuration, the patch levels, or the application composition. For a web application, vulnerability scans may require access to user credentials to scan the flow of an application according to how users interact with the application. Vulnerability scans should span the entire application flow, across the whole application, the stack, and all supporting platforms. 

- Some tools available for vulnerability scanning are Coverity, CodeSonar, Snyk Code, and Static Reviewer. They are examples of static application security testing (or SAST) tools: 
    - Coverity is an incremental analysis scanner for programming languages such as C, C++, Java, and Python 
    
    - CodeSonar uses abstraction to model the code and find any weaknesses in paths and program variables. Snyk Code is an integrated development tool that performs semantic analysis to discover coding and security bugs throughout the development phase

    - Static Reviewer eliminates well-known vulnerabilities 

## Threat Modeling

- Threat modeling is identifying, categorizing, and enumerating security threats. Threat modeling provides a process to analyze ongoing threats and eliminate the potential for software coding weaknesses and vulnerabilities.

- Threat models use diagrams to represent data flows within software applications. Where does threat modeling belong in the software development lifecycle (or SDLC)? The best time is during the design phase. 

    <p align="center">
    <img src="./assets/application_security/thread-modeling.png" alt="drawing" width="600" height="300" style="center" />
    </p>

- Three popular threat models that you can use are: 
    - Process for Attack Simulation and Threat Analysis (or PASTA)
    - Visual, Agile, and Simple Threat (or VAST)
    - STRIDE

    <p align="center">
    <img src="./assets/application_security/thread-models.png" alt="drawing" width="600" height="300" style="center" />
    </p>


## Threat Monitoring

- Threat monitoring is scanning code repositories and containers to find security issues such as Password mishandling, protocol insecurities, and incorrect permissions

- Where does threat modeling belong in the software development lifecycle (or SDLC)? Actually, you integrate threat modeling the Develop stage, the Test stage, and the Deploy stage

- Using code scanning in integrated development environments (or IDEs) and source control management (or SCM) tools supports the SDLC by integrating security checks from development to deployment. Code scanning tools reference databases that store security threats and vulnerabilities such as the Open Web Application Security Project (or OWASP) Top 10

- To perform threat monitoring, you can use code checker tools. A code checker scans source code for any security issues and vulnerabilities, which will alert you to coding problems. Code checkers analyze code to find issues with coding syntax, style, and documentation as well as insights into where to fix issues in the code. So, using a code checker helps you develop secure code and improve quality in your application. 

- You can integrate threat monitoring into your code repositories. Because repositories are often collaborative and open source, they carry a significant risk of security threats and vulnerabilities. Integrating threat monitoring with code repositories enables code scanning of source code management tools such as GitHub. You can leverage code project monitoring that can generate automatic “fix” pull requests while scanning code repositories. Code scanners provide vulnerability reporting and insights after they scan code in your repositories. They also scan and test every pull request for security vulnerabilities. And sign commits with a public encryption or pretty good privacy (PGP) key as verification of trusted sources. 

- Another type of threat monitoring is _container scanning_, which is the process of scanning container images that contain code. Containers are packages of application code and their packaged library dependencies. Because containers have dependencies, they are exposed to security vulnerabilities from external sources. Container scanning scans code deployed to containers, which may contain vulnerabilities and security threats. Because container images are usually built from other container images that may have vulnerabilities, container scanning must include not only the base image but all the other layered container images as well. Monitoring all container images helps reduce security risks.

## Introduction to Nmap
- Nmap, short for Network Mapper, is an open-source network scanning and security auditing tool. It is used to detect and fingerprint network devices, services, and operating systems, as well as to detect security vulnerabilities. It can be used to scan the network for open ports, detect operating systems, and detect vulnerabilities such as buffer overflows, intrusions, and denial-of-service attacks. 

- Nmap is commonly used in security audits, penetration tests, and system administration. It was developed by Gordon Lyon, widely known by his pseudonym ‘Fyodor,’ and was first released in September 1997. Nmap is designed to discover hosts and services on a computer network, thus creating a map of the network's structure. This tool has gained immense popularity in the field of network security due to its versatility and extensive capabilities.

- Nmap is a versatile and widely used tool in the field of cybersecurity, catering to various roles and purposes within the cybersecurity and IT communities. Types of Nmap Scans with Examples:

    - TCP Connect Scan (Default Scan): Basic scan that opens a full TCP connection to each target port. Example: `nmap -sT target`

    - SYN Stealth Scan: Also known as a half-open scan, it sends SYN packets and analyzes responses. Example: `nmap -sS target`

    - UDP Scan: Sends UDP packets to target ports to identify open UDP services. Example: `nmap -sU target`

    - ACK Scan: Sends TCP ACK packets to determine firewall configurations. Example: `nmap -sA target`

    - Version Detection (-sV): Identifies service versions running on open ports.
    Example: `nmap -sV target`

    - OS Detection (-O): Attempts to identify the target's operating system.
    Example: `nmap -O target`

    - Script Scanning (-sC): Executes predefined scripts to gather additional information. Example: `nmap -sC target`

    - Ping Scans: Various ping techniques to check target's availability.
    Example: `nmap -PE target` (ICMP Echo Request)

    - Traceroute (–traceroute): Performs traceroute to determine the path packets take. Example: `nmap --traceroute target`

    - TCP Null Scan: Sends packets with no TCP flags set to observe responses.
    Example: `nmap -sN target`

    - TCP FIN Scan: Sends packets with FIN flag set to observe responses.
    Example: `nmap -sF target`

    - TCP Xmas Scan: Sends packets with various TCP flags set to observe responses.
    Example: `nmap -sX target`

    The choice of scan depends on what kind of information you're looking for and the level of visibility you require.


## Security Testing

Security tests are procedures for comparing the states of an application or a system. Security testing provides a secure code baseline for development. You should perform security tests on all new code to reduce the risk of impacts. Any code changes may create vulnerabilities in previously secure code. 

Secure testing takes place during the Test stage along with code review. Although secure code should be a top priority during the Test phase, security testing should be part of your secure coding processes throughout the SDLC. 

To perform security testing, the first step is to provide a secure baseline during development. Once a baseline has been established, you can compare the states of an application or a system. Functional security testing should be an integral part of your security testing processes. Functional security testing is the expectation of behavior when you test software or a system. To perform functional security testing, you need a list of your functional requirements. Functional security testing helps you ensure that the code meets your security requirements. 

Two types of functional testing are: Ad hoc testing. And exploratory testing. Ad hoc testing is specialized testing upon discovery of a vulnerability. Exploratory testing takes place outside of formal testing. Examples of exploratory testing are testing a theory or testing an idea. In automated security testing, two popular testing procedures are unit testing and integration testing. Unit tests are for testing classes and methods to evaluate application programming interface (or API) contracts. You can perform unit testing on individual classes with limited scope. Integration tests are for testing the integration of several code classes within an application. You can perform integration tests across application tiers and a wide testing scope. You can also use automated frameworks for automating security tests of an application or system. 

Three examples of security testing automation frameworks are BDD-Security, Mittn, and Guantlt. 

BDD-Security is a security testing framework that uses behavior-driven development. Mittn is a popular tool suite to include in continuous integration. And Gauntlt is a security framework that hooks into security tools for simplified integration. Using mitigation strategies helps reduce risks and impacts of security threats and vulnerabilities. As you develop code, use these five key mitigation strategies. First, use JavaScript Object Notation (or JSON) for your API data payloads. Unlike Extensible Markup Language (or XML), JSON allows simplistic data encoding in key-value pairs instead of complex nested elements and slower parsing. Next, implement secure coding practices. Communicate security standards across your team and within your organization. Use vulnerability scanners to find vulnerabilities in code. You can also automate vulnerability scanning. 


# Static Analysis
Static analysis examines all code or runtime binaries to help detect common vulnerabilities. It's a debugging method that automatically inspects source code before execution. Static application security testing (or SAST) examines source code to identify security flaws that render your organization's applications vulnerable to attack. It is extremely effective in detecting problems in code and do not require code to be complete. 

Static analysis can take a long time because it thoroughly scans the code. Where does static analysis belong in the software development lifecycle (or SDLC)? Static code analysis takes place early in the development process before software testing begins. For DevOps enterprises, static analysis occurs during the Develop stage And establishes an automatic feedback loop. So, you will become aware of any issues with your code from the start. And it will be simpler for you to resolve those issues. 

Static Application Security Testing (SAST) is SAST is also known as white-box testing as it involves testing the internal structure and workings of an application. This helps prevent security breaches and minimizes the risk of costly security incidents. One of the primary benefits of SAST is that it can identify vulnerabilities that may not be detected by other testing methods such as dynamic testing or manual testing. This is because SAST analyzes the entire codebase. There are several types of vulnerabilities that SAST can identify, including:

- Input validation vulnerabilities: These vulnerabilities occur when an application does not adequately validate user input, allowing attackers to input malicious code or data that can compromise the security of the application
- Cross-site scripting (XSS) vulnerabilities: These vulnerabilities allow attackers to inject malicious scripts into web applications, allowing them to steal sensitive information or manipulate the application for their own gain.
- Injection vulnerabilities: These vulnerabilities allow attackers to inject malicious code or data into the application, allowing them to gain unauthorized access to sensitive information or execute unauthorized actions.
- Unsafe functions and libraries: These vulnerabilities occur when an application uses unsafe functions or libraries that can be exploited by attackers.
- Security misconfigurations: These vulnerabilities occur when an application is not properly configured, allowing attackers to gain access to sensitive information or execute unauthorized actions.

SAST tools can be added into your IDE which helps you detect issues during software development and can save time and effort, especially when compared to finding vulnerabilities later in the development cycle. You can use **SonarQube**, an open-source platform built by SonarSource for continuous code quality assessment. It can integrate with your existing workflow to enable continuous code inspection across your project branches and pull requests.

SAST Tools (with free tier plan):

- SonarCloud: SonarCloud is a cloud-based code analysis service designed to detect code quality issues in 25+ different programming languages, continuously ensuring the maintainability, reliability and security of your code.
- Snyk: Snyk is a platform allowing you to scan, prioritize, and fix security vulnerabilities in your own code, open source dependencies, container images, and Infrastructure as Code (IaC) configurations.
- Semgrep: Semgrep is a fast, open source, static analysis engine for finding bugs, detecting dependency vulnerabilities, and enforcing code standards.

How SAST Works?

Tools for static analysis examine your code statically on the file system in a non-runtime environment. This practice is also known as source code analysis. The best static code analysis tools offer depth, speed, and accuracy. SAST tools typically use a variety of techniques to analyze the source code, including:

- Pattern matching: involves looking for specific patterns in the code that may indicate a vulnerability, such as the use of a known vulnerable library or the execution of user input without proper sanitization.

- Rule-based analysis: involves the use of a set of predefined rules to identify potential vulnerabilities, such as the use of weak cryptography or the lack of input validation.

- Data flow analysis: involves tracking the flow of data through the application and identifying potential vulnerabilities that may arise as a result, such as the handling of sensitive data in an insecure manner.

Consideration while using SAST Tools

- It is important to ensure that the tool is properly configured and that it is being used in a way that is consistent with best practices. This may include setting the tool's sensitivity level to ensure that it is properly identifying vulnerabilities, as well as configuring the tool to ignore certain types of vulnerabilities that are known to be benign.

- SAST tools are not a replacement for manual code review. While these tools can identify many potential vulnerabilities, they may not be able to identify all of them, and it is important for developers to manually review the code to ensure that it is secure.

Challenges associated with SAST

- False positives: Automated SAST tools can sometimes identify potential vulnerabilities that are not actually vulnerabilities. This can lead to a large number of false positives that need to be manually reviewed, increasing the time and cost of the testing process.
- Limited coverage: SAST can only identify vulnerabilities in the source code that is analyzed. If an application uses external libraries or APIs, these may not be covered by the SAST process.
- Code complexity: SAST can be more challenging for larger codebases or codebases that are written in languages that are difficult to analyze.
- Limited testing: SAST does not execute the code and therefore cannot identify vulnerabilities that may only occur when the code is executed.

Despite these challenges, SAST is a valuable method of evaluating the security of an application and can help organizations prevent security breaches and minimize the risk of costly security incidents. By identifying and fixing vulnerabilities early in the SDLC, organizations can build more secure applications and improve the overall security of their systems.

**Implementing _SonarCloud_ into your CI workflow:
See [here](https://github.com/MichaelCade/90DaysOfDevOps/blob/main/2023/day09.md) for details about implementing SonarCloud into your CI workflow using Github Actions.**

Ref:

[SAST – All About Static Application Security Testing](https://www.mend.io/blog/sast-static-application-security-testing/)


## Dynamic Analysis

Dynamic analysis is the process of testing and evaluating an application as it is executing. Dynamic analysis is typically run against fully built applications. So it is performed in staging, pre-prod. DAST should never be run against production environment! because it might be harmful to backend database or compromising credential because all DAST tools have the possibility for authentication into the application which could lead to production credentials compromise. . It helps identify vulnerabilities, memory issues, and possible crashes in production. 

Dynamic application security testing (or DAST) evaluates the application from the outside in through the front end. DAST acts like an attacker. It simulates attacks from external sources to detect potential threats and vulnerabilities. Because DAST does not have access to the source code, it performs black-box testing, analyzing behaviors of inputs and outputs. Idea is to automate as much as possible black-box penetration testing. It uses pretty standardized attacks, external controls can block the attacking traffic and this way to cover potentially exploitable flows (as per definition adversary would be able to eventually bypass such measures)

It is an essential component of the SSDLC. By conducting DAST testing, organizations can prevent security incidents and protect their data and assets from being compromised by attackers. Here are three key benefits of using dynamic analysis:

- A dynamic analysis tool crawls an application's interface in a dynamic manner. It performs tests that provide insights. And it helps flush out faults in dynamic code paths. Using dynamic scanning, the dynamic analysis tool examines the code from the root URL. 

- With dynamic scanning, you can intentionally try to break into your system. And you can get the upper hand by identifying breakpoints and vulnerabilities in your code to patch up. So, that’s one benefit. 

- Next, a dynamic analysis tool performs tests that show how your application reacts to various inputs. Inputs can be in the form of an action made by a URL or a form. Here you need to make sure that the data you are trying to use is not harming your real-time database in any manner, so for safety, use a dummy database for running your tests. 


Common error is scanning compensating security controls (e.g. WAF) instead of the real application. DAST is in its core an application security testing tool and should be used against actual applications, not against security mitigations. As  Actual scans are quite slow, so sometimes they should be run outside of the DevOps pipeline. Good example is running them nightly or during the weekend. Some of the simple tools (zap / arachny, …) could be used into pipelines but often, due to the nature of the scan can slow down the whole development process.

Once the DAST testing is complete, the results are analyzed to identify any vulnerabilities that were discovered. The organization can then take appropriate remediation steps to address the vulnerabilities and improve the overall security of the application. This may involve fixing the underlying code, implementing additional security controls, such as input validation and filtering, or both. 

Open-source tools, such as ZAP, Burp Suite, and Arachni, can be used to conduct DAST testing and help organizations improve their overall security posture. As with all other tools part of DevSecOps pipeline DAST should not be the only scanner in place and as with all others, it is not a substitute for penetration test and good development practices.

Some useful links and open-source tools: 
[zaproxy](https://github.com/zaproxy/zaproxy)
, [arachni](https://www.arachni-scanner.com/)
, [owasp](https://owasp.org/www-project-devsecops-guideline/latest/02b-Dynamic-Application-Security-Testing)


See [this](https://github.com/MichaelCade/90DaysOfDevOps/blob/main/2023/day20.md) for a example of using DAST with ZAP Proxy for causong some damage.

Below is a summary of the some warnings that might come back after applying ZAP to an application URL. Each describes the vulnerability, then cites the number of times it was found (for example, x 3), and then lists the URLs that had the vulnerability.

```
WARN-NEW: Re-examine Cache-control Directives [10015] x 3
WARN-NEW: Cross-Domain JavaScript Source File Inclusion [10017] x 4
WARN-NEW: Missing Anti-clickjacking Header [10020] x 2
WARN-NEW: X-Content-Type-Options Header Missing [10021] x 9
WARN-NEW: Information Disclosure - Suspicious Comments [10027] x 2
WARN-NEW: Server Leaks Information via "X-Powered-By" HTTP Response Header Field(s) [10037] x WARN-NEW: Content Security Policy (CSP) Header Not Set [10038] x 2
WARN-NEW: Timestamp Disclosure - Unix [10096] x 4
WARN-NEW: Cross-Domain Misconfiguration [10098] x 9
WARN-NEW: Loosely Scoped Cookie [90033] x 3
FAIL-NEW: 0     FAIL-INPROG: 0  WARN-NEW: 10    WARN-INPROG: 0  INFO: 0 IGNORE: 0       PASS: 24

```
The above result shows that this application has vulnerabilities in Cross-Domain JavaScript Source File Inclusion, Missing Anti-clickjacking Header, X-Content-Type-Options Header Missing, Content Security Policy (CSP) Header Not Set, Cross-Domain Misconfiguration, and Loosely Scoped Cookies, just to name a few.
You can use the numbers next to the vulnerability names to read about the alert on the ZAP Proxy Web site. Using the following URL:

`https://www.zaproxy.org/docs/alerts/{NUMBER}`

Be sure to replace {NUMBER} with the number of the alert.
As a developer, your task would be to look up the vulnerability, look at each URL listed as being vulnerable, and then fix the vulnerabilities in the code one by one.


## Vulnerability Analysis

Vulnerability analysis is a method of identifying possible application flaws that could jeopardize your application. Vulnerabilities in your code occur every day. Say you scanned your code one day and found no vulnerabilities. But it had a vulnerability that you didn't know about. Or a library that you use has a new vulnerability that was just discovered. And that’s how code that was secure yesterday becomes susceptible today. Anyone could break into your system if they found a loophole by trying a few things here and there. When you perform your security checks and scans and add nothing to the code, it can still become susceptible simply because someone discovered some minor flaw, fault, or situation that can be exploited. Some programs and platforms can help you scan for vulnerabilities in your code. And vulnerability reports are published daily. 

If a new report says that the library or plugin you are using is a vulnerable version, you might want to update to a new version to prevent your application from being attacked. One example of such a platform is Snyk, which is a developer security platform for securing code, dependencies, containers, and infrastructure as code. And you may use one of these three vulnerability tools: Burp Suite, Nessus, and Zed Attack Proxy. 

 <p align="center">
<img src="./assets/application_security/vulnaribility-analysis.png" alt="drawing" width="600" height="300" style="center" />
</p>

 - _Burp Suite_ is a vulnerability scanner that is popular for scanning web applications. You can set up automated scans of a website or perform manual scanning by crawling the overall structure of a website or web application.By running multiple audit phases, Burp Suite can find and exploit functions involving user input. Burp Suite audits vulnerabilities in three phases: passive, active, and JavaScript analysis. 
 
 - The next notable vulnerability tool is _Nessus_. Nessus is a well-known vulnerability scanner that works on Mac, Linux, and Windows. You can install and run it as a local web application. Nessus provides a scripting language to write specific tests and plugins to detect a particular vulnerability or a common virus. 
 
 - Another vulnerability scanner is an OWASP tool called Zed Attack Proxy (or Zap). Zap is an open source software that uses spiders to crawl web applications. Zap actively or passively scans HTML files in web applications via links and AJAX applications via generated links. 
 
 - Three best practices that you can follow to prevent vulnerabilities in your code are: 
    - Training your development team on security. This is the single most important thing to do and taking courses that are designed specifically for developers goes a long way toward making your software more secure. 
    - Performing policy analysis and updating, so that your policies are always appropriate and up to date. And automating your process for vulnerability analysis so that even if your developers forget to scan, your automation will scan for them and alert you quickly of any new vulnerabilities. 


## Runtime Protection

Runtime protection is a modern security mechanism that shields applications against threats while the applications are running. How can you achieve runtime protection? Use security tools that can scan applications for weaknesses while the applications are running. 

Interactive Application Self-testing (or IAST) scans for vulnerabilities during the testing process. And Runtime Application Self-Protection (or RASP) looks for assaults in the production environment. During testing, when you implement IAST, it detects security flaws in real time before the application is released to the public. You get critical information about where to find the problem and fix it fast before it can cause data breaches. And you can run IAST along with other automated testing procedures. 

So, what are some features of IAST? IAST produces low false-positive output from examining your application in real time. Integrating IAST into continuous integration/continuous delivery (or CI/CD) is simple. It can connect smoothly with standard build, test, and quality assurance tools, and you don’t need to perform much configuration or tuning to reduce false positives. IAST enables earlier and less expensive fixes. 

When you have deployed your application and it is running in production, RASP can provide runtime level protection in the production environment. Integrated into an application, RASP safeguards software from harmful inputs by assessing the program's behavior and the context of the activity. RASP helps identify and prevent assaults in real time without requiring human involvement. As it monitors the application, It observes and assesses the activity continuously. Next, what are some features of RASP? RASP protects against exploitation. It intercepts all types of traffic that could indicate malicious activity, including structured query language (or SQL) injection, exploits, and bots. 

When RASP senses a threat, it can terminate the user's session and notify the security team. RASP can work directly within an application. It is simple to deploy, and it is inherently capable of monitoring and self-protecting application behavior. RASP detects and prevents attacks with great precision, separating malicious requests from legal requests, and minimizing false positives. You can incorporate RASP into different DevOps systems. Securing the cloud is not an easy task and requires much effort because applications are running on someone else's infrastructure outside your secure network. Luckily, RASP is extremely compatible with cloud computing.



## Software Composition Analysis (SCA): Knowing what’s inside your apps

- Securing your business from potential issues and vulnerabilities in open source development is becoming increasingly important. Software component analysis (or SCA) is the process of determining which open source components and dependencies are used in your application. You can use SCA tools throughout your software development workflow to ensure that any imported libraries or dependencies do not cause security risks or legal compliance issues in your code. 

-SCA tools scan the codebase of a software project and provide a report that lists all the open source libraries, frameworks, and components that are being used. This report includes information about the licenses and vulnerabilities of these open source libraries and components, as well as any security risks that may be associated with them.

- SCA looks for all the dependencies linked to your code, including some that you might not be aware of. For example, if you are importing Flask, Flask may require and install dependencies that you may not need. Even if you are using a version that is not vulnerable, one of the dependencies that Flask is using might be vulnerable. 

- If you are working for an enterprise, you must ensure that the libraries you use do not contain a GNU General Public License (or simply a GPL License). If you link to a GPL License library, you must be willing to give away your source code. That won’t be a problem if you are in an open source environment. But if you do not want to give away the source code of your product, you will be in trouble! because you will be giving away all of your classified information. Overall, SCA gives developers visibility into and control of potential security flaws in the open source components that they use. 

Here are four goals of SCA:

<p align="center">
<img src="./assets/application_security/sca-goals.png" alt="drawing" width="600" height="300" style="center" />
</p>

- All open source components should be discovered and tracked
- Open source license compliance should be tracked to reduce risk 
- Open source vulnerabilities should be identified
- A variety of scans should be run, depending on the situation and requirements

Three industry efforts to identify software components are: The National Institute of Standards and Technology (or NIST) CPE Dictionary, which is a centralized database for common platform enumeration (or CPE) of products, Software Identification Tags (or SWID Tags), are a standard to describe commercial software, and package URL specification. 

An example of a package URL specification is a string that starts with scheme followed by type slash namespace slash name at version question mark qualifiers hashtag the subpath. To verify software components, follow industry standards. Two standards that you can use are: 

- OWASP Software Component Verification Standard, which is a community-supported effort to build a sustainable framework for reducing risk within a software supply chain
- Supply-chain Levels for Software Artifacts (or “salsa”), which provides a security framework for improving integrity and preventing tampering by implementing standards and controls 

Consider using these four popular SCA tools: 

- **GitHub SCA** is for viewing dependency packages and vulnerabilities while using GitHub.com 
- Two SCA tools that OWASP offers are _Dependency-Check_ and _Dependency-Track_. Dependency Check is for checking for vulnerabilities within project dependencies. And Dependency-Track is for identifying any risks within the software supply chain
- **Snyk** is for analyzing codebases to evaluate security, code quality, and licensing. Snyk Open Source provides a developer-first SCA solution, helping developers find, prioritize, and fix security vulnerabilities and license issues in open source dependencies


<p align="center">
<img src="./assets/application_security/sca-tools.png" alt="drawing" width="600" height="300" style="center" />
</p>

SCA tools can identify several risk factors, including:
- Outdated components
- Components with known vulnerabilities
- Component quality: from a security standpoint, a component might be considered lower quality if it is poorly maintained or has a very small community supporting it.
- Transitive dependencies: SCA tools can track vulnerabilities from transitive dependencies. When a component relies upon another component, that dependency is referred to as transitive.
- External services: A component may interact with external services, such as Web APIs. SCA tools can identify when this interaction might be a vulnerability.

Software Composition Analysis offers two chief benefits:

- Detection of third-party software components that introduce vulnerabilities or other security risks into an application. Knowing which specific parts of the application create risks makes it easier for developers to identify and mitigate supply chain security threats

- Identification of third-party software components that may be subject to specific licensing terms, like those defined in a particular open source license. This is important because failure to adhere to those licensing terms could lead to legal and compliance risks for the organization.


### OWASP Dependency Check

Dependency-Check is a Software Composition Analysis (SCA) tool that attempts to detect publicly disclosed vulnerabilities contained within a project's dependencies. Dependencies are the software components that your code relies on for additional functionality. The SCA tool will generate a report listing the dependency, any identified Common Platform Enumeration (CPE) identifiers, and the associated Common Vulnerability and Exposure (CVE) entries.

To use OWASP Dependency Check, you will need to include it as a part of your build process. There are integrations available for a variety of build tools, including Maven, Gradle, and Ant. You can also use the command-line interface to scan your dependencies.

See [here](https://github.com/MichaelCade/90DaysOfDevOps/blob/main/2023/day11.md) for integrating Dependency Check with GitHub Actions

References:
[Dependency Check Documents](https://jeremylong.github.io/DependencyCheck/), [Source Code](https://github.com/jeremylong/DependencyCheck)




## Code Practices

Code practices are part of the software development process for the development of secure software. Security is a major concern in the DevOps community because attackers target insecure code in the application layer. Implementing code practices is an important part of developing secure software and when implemented early in development, is cost-effective because correcting unsecure code later in the software development process is expensive.

Here are some general code practices you should follow when developing software: 
- Implement a secure software development lifecycle. Including security in the development lifecycle is cost-effective and ensures your application is as secure as it can be, right from the start 
- Establish secure coding standards. Following a set of secure coding standards establishes good habits
- Build and use reusable object libraries for efficiency and to reduce risk
- Develop with only tested and approved managed code
- Implement safe updating by focusing on exposed threats or source code that contains security-critical components
- Attend training courses that focus on secure software development.They can increase your security awareness and strengthen your skills. 

### Input Validation

Validating input means to check (on the server side), that the input provided by user or attacker is what you expect it to be. What should you validate? Any input data you use that a hacker can manipulate. 

Check your input data for: 
- Expected data types, Data range, and data length
- Allowed characters against a "white" list
- If the input data isn’t the right type, it should be rejected
- Any data coming from untrusted sources should also be validated
- Reduce any additional risk by developing on trusted and hardened systems only
- Whitelist validation should always be performed for allowed characters 
- Scrub or remove any malicious characters if entered as input data. Malicious characters may include the following: < > " ' %  ( ) & + \ \ ‘ \ “  Anything that the attacker can use to make your application do something it wasn’t intended to do. If any of these malicious characters are actually allowed as valid input, you should implement additional controls such as 
    - output encoding
    - securing task-specific APIs 
    - accounting for all data input throughout an entire application

### Output Encoding

Output encoding is 
- the translation of input code to safe output code
- Implement a policy and practice for each type of outbound encoding used
- Encode all characters unless any are unsafe for the interpreter
- Sanitize all output of untrusted queries such as SQL, XML, and LDAP
- Sanitize all untrusted data output to local operating system commands 


### Error Handling and Logging

- Improper error handling can expose a variety of security risks for an application
- Meaningful error messages and logging: error messages containing too much detail provide attackers with valuable clues about potential flaws they can exploit. Provide diagnostic information for troubleshooting, and provide no useful information to an attacker
Use custom error pages and generic messages for error handling and logging
- Release allocated memory when any error conditions occur to avoid corruption
- Implement access restrictions to logs to keep attackers out 
- You should also log any types of tampering events and failures such as input, authentication attempts, and access control

### Dependencies

A dependency is needed when a piece of software or code relies on another to function. They’re commonly used to add features and functionality to software without writing it from scratch. Dependencies are reusable code found in a library (package or module) that your code makes calls to. You can use a package manager to automate the download and installation of dependencies. 

Here are some of the benefits of using dependencies in your code: 
- The software development process becomes faster and more efficient 
- You can deliver software more quickly by building on previous work
- Dependencies enable applications to have more features and functionality 
- Functionalities provided by the dependency eliminate having to write it from scratch. The functionality provided by the dependency could perform better than the native implementation. 

Dependencies challenges and risks:

- Downloading and using code from the Internet is risky. It could expose your software to vulnerabilities, bugs, or other flaws 
- Production risk could occur as a result of implementing incompatible, outdated, or missing dependencies: 
    - Performance degradation or crashes. 
    - Data could be leaked so your company’s reputation could also be impacted, resulting in loss of business, reputation or even fines 
- Licensing challenges: Be aware of any license requirements for dependencies you use. Use the correct type of licensing for your project. Make sure there's no unlicensed code in your application. 

If you plan to use dependencies in your project, it's best practice to vet (or examine) them thoroughly before implementation. Vet the dependency by checking the following:
- Design: Check that the API is well-designed and well-documented 
- Quality: Check the quality of the code for undesired behavior, and semantic problems
- Testing: Test the basic code functionality and look for any possible failures
- Debugging: Check dependency's issue tracker for open issues and bug reports
- Maintenance: Review the commit history for bug fixes and ongoing improvements. Avoid using dependencies that haven't been updated for more than a year 
- Usage: Is the dependency widely adopted or seldom used? Seldom-used dependencies could be abandoned
- Security: Software dependencies can present a large surface for attacks. Look for weaknesses and vulnerabilities that allow malicious input
- Use dependency management tools to manage downloads, track version updates. 


Dependencies’s dependency

A dependency that relies on another dependency isn't bad; however, it does pose some challenges. Code problems found within indirect dependencies may have an impact on your code. So, you should inspect all indirect dependencies. Use a dependency manager to list any direct and indirect dependencies for inspecting all code. When you upgrade dependencies, be aware of any new, indirect dependencies that could also make their way into your project. 


### Secure Development Environment

Apps developed without security input from Ops are highly susceptible to cyber-attacks. When you factor in the cost of downtime, the cost to fix the security vulnerabilities, the possibility of a data breach or leak, and the reputation of your organization – it becomes clear that implementing security in application development should not be a last-minute decision. Prevent this from happening in your organization by involving the security team early in the software development process. Get some solid collaboration with the security people early in the design phase. It’s critical to begin securely writing your code from the start. 

But writing secure code isn’t enough. The development environment must be secure, too. Development systems and platforms are also vulnerable to the same types of attacks as production machines. You must harden the environment to keep threat actors out. For that, we rely on the Security team. 

Let’s understand how to develop a secure application with DevOps. 

- If the environment isn’t secure, it's difficult to accept that the code coming from it is also secure. Security is a team effort. Everyone on the DevOps team is responsible for security. What is secure development environment? It is an ongoing process of securing the network, compute resources, and storage devices both on-premise and in the cloud. 

Securing your development environment reduces the risk of an attacker who tries to: 
- Steal sensitive info such as encryption and access keys, or intellectual property 
- Embed malicious code into your project without your knowledge
- Use your system as a tool to launch other attacks into your build and deployment pipeline or other machines on the network. 

The process entails: Keeping all software up-to-date and Removing or disabling unnecessary services. It's important to physically securing development machines and use separate machines for coding and business: 
- Use a virtual machine, a Docker container, or a separate computer for business-related functions and develop your code on a hardened system to reduce phishing, malware attacks, and other cyber threats. 
I do all of my development in Docker containers so that I have an isolated known environment every time I begin to code. And my project repositories are set up so that every member of my team has the same containerized environment as well (DevContainers). 
- Use complex passwords and frequent password changes, and implement multifactor authentication. 
- Protect the code repository and secure your build and development pipeline. 
- Invest in monitoring, logging, and auditing controls. and continually test for security and plan for security flaws. 

Finally, your development machines and any code on them could be vulnerable if there is no up-to-date antivirus or anti-malware products installed on your development systems, leaving them vulnerable to phishing, malware, and other attacks. Unrestricted access to unapproved code repositories and lack of governance or policies for obtaining code could allow suspect software dependencies into your application. 


Here are some best practices you can implement to secure your development environment:
- Secure the internet connection. Insecure networks are highly vulnerable to network attacks
- Achieve a secure internet connection by: regularly checking for open ports and closing any ports not needed 
- Set up firewalls with strict ingress and egress traffic policies to ensure nothing other than allowed traffic is granted. This is where developing in Docker containers is really helpful because the containers are on a separate isolated network from your development computer, and all ports are closed to the outside by default. 
- Implement multifactor authentication to protect against identity theft. Passwords alone aren’t enough. Relying solely on passwords leaves your system at high risk of being attacked. Also, if the password is traced, your entire code is at risk – along with other assets. Multifactor authentication also prevents the attacker from leveraging the permissions of a developer. You can also secure secrets with multifactor authentication. And it protects them from being stolen and reduces the risk of losing them. 
- Add additional security for those developers who need to access production systems from their developer machines. You should monitor developers environments but monitoring developers’ environments and activities don’t mean keeping an eye on everything they do. Developer machines should be locked down as tightly as possible yet still permit access to the required resources to get the job done. Trust me on this one, otherwise, developers will start using ‘workarounds’ to defeat security checks, leaving their machines vulnerable to attacks. - Incorporate daily security habits and behavior. 
You should watch for suspicious activity and use network utilities to check whether websites visited are suspicious or safe. 

- Track all commits, and changes made by developers in the environment for future reference in case anything goes wrong. And using pre-commit hooks, you can even check to make sure that developers aren’t checking in sensitive data like credentials to their code repositories.


## Git Secret Scan
- Prevent pushing secret to GitHub repository by enabling Push protection for your GitHub accounts. Once enabled, all the commits are scanned for secrets and blocked if secrets found in which case you need to resolve that push
- AWS has `git-secrets` tool that prevents you from committing sensitive information into Git repositories.`git-secrets` scans commits, commit messages, and merges to prevent sensitive information such as secrets from being added to your Git repositories. For example, if a commit, commit message, or any commit in a merge history matches one of your configured, prohibited regular expression patterns, the commit is rejected.


    Install it on your local machine:

    ```
    git clone https://github.com/awslabs/git-secrets.git
    make install
    ```
    Then, cd to your repository and scan it:
    ```
    cd my-git-repository
    # configure it to scan your Git repository on each commit
    git secrets --register-aws
    # start scanning
    git secrets -–scan 
    ```

    [Reference](https://docs.aws.amazon.com/prescriptive-guidance/latest/patterns/scan-git-repositories-for-sensitive-information-and-security-issues-by-using-git-secrets.html)
    

Useful Linkes:
- [90DaysDevOps](https://github.com/MichaelCade/90DaysOfDevOps/tree/main)


## Continuous Image Repository Scan

A container image consists of an image manifest, a filesystem and an image configuration. 1

For example, the filesystem of a container image for a Java application will have a Linux filesystem, the JVM, and the JAR/WAR file that represents our application.

If we are working with containers, an important part of our CI/CD pipeline should be the process of scanning these containers for known vulnerabilities. This can give us valuable information about the number of vulnerabilities we have inside our containers, and can help us prevent deploying vulnerable applications to our production environment, and being hacked because of these vulnerabilities.

The image scanning process consists of looking inside the container, getting the list of installed packages (that could be Linux packages, but also Java, Go, JavaScript packages, etc.), cross-referencing the package list against a database of known vulnerabilities for each package, and in the end producing a list of vulnerabilities for the given container image.

There are many open-source and proprietary image scanners, that you can install and start scanning your container images right away, either locally of your machine or in your CI/CD pipeline. Two of the most popular ones are Trivy and Grype. Scanning a container image is as simple as installing one of these and running them against an image:
```
grype ubuntu:latest
```
If an image scanner tells you that you have 0 vulnerabilities in your image, that does not mean that you are 100% secure. Scanning a container image with Grype is as simple as running `grype <image>`

Enforce a set of rules for our container images. For example, a good rule would be "an image should not have critical vulnerabilities" or "an image should not have vulnerabilities with available fixes." Fortunately for us, this is also something that Grype supports out of the box. We can use the `--fail-on <SEVERITY>` flag to tell Grype to exit with a non-zero exit code if, during the scan, it found vulnerabilities with a severity higher or equal to the one we specified. This will fail our pipeline, and the engineer would have to look at the results and fix something in order to make it pass. Example:

```
grype springio/petclinic:latest --fail-on critical
```

Sometimes a vulnerability we encounter will not have a fix available. These are so-called zero-day vulnerabilities that are disclosed before a fix is available. Also, it is not dangerous! In this case, we can tell Grype to ignore this vulnerability and not fail the scan because of it. We can do this via the `grype.yaml` configuration file, where we can list vulnerabilities we want to ignore:
```
ignore:
  # This is the full set of supported rule fields:
  - vulnerability: CVE-2016-1000027
    fix-state: unknown
    package:
      name: spring-core
      version: 5.3.6
      type: java-archive
  # We can list as many of these as we want
  - vulnerability: CVE-2022-22965
  # Or list whole packages which we want to ignore
  - package:
      type: gem
```

### SBOM
SBOM stands for Software Bill Of Materials.

It is a list of all the components that make up a software application or system. It includes information about the various third-party libraries, frameworks, and other open-source or proprietary components that are used to build the software. An SBOM can also include details about the versions of these components, their licensing information, and any known vulnerabilities or security issues.

The objective of an SBOM is to list these components, providing software users visibility over what is included in a software product, and allowing them to avoid components that can be harmful for security or legal reasons.

In the context of a container image, an SBOM for a container image will contain:

- Linux packages and libraries installed in the containers
- the language-specific packages installed for the application running in the container (e.g. Python packages, Go packages, etc.)

There are tool that can help you extract the SBOM from a container images.

One such tool is `syft`.

For example, we can use syft to generate the SBOM for the ubuntu:latest container image:

`syft ubuntu`

We see that the SBOM not only contains the packages and libraries installed inside the container image, but also list their types and versions. We can use now cross-reference this list with a vulnerability database to see whether we have any vulnerabilities inside the container.

# Rough Design of a CI/CD Pipeline
- Every pipeline starts with a source stage. This is ususally a code repository: 
    - As a branch protection rule,  make it is necssary to do a PR review before developers can merge their code to the branch
    - Apply linting to check your code for syntax errors. Any language has a linting program. Implement linting into your CI/CD pipeline. GitHub Actions is a good choice. Fail a PR if linting conditions not resolved. It is better for linting to happend on IDE even before developer commit their code. To do this, create a precommit hook in IDE that runs these linting checks before any commit (local precommit checks)
- Build: 
    - Compiling your code and building images to be ready for testing
    - Run unit tests with code coverage 80-90%
- Test: heavy testing happens here
    - Integration Test: tests functionality of your application (BDD)
- Release: ship your image to the registry
- Create staging (QA or pre-production) and perform other testing in a production-like environment
- Final review and approval
- Deploy

# Example of Implementations of a CI/CD Pipeline

<p align="center">
<img src="./assets/application_security/CICD-pipeline.png" alt="drawing" width="600" height="300" style="center" />
</p>

- Install Jenkins on virtual machine to be the Jenkins server: 
    - Install Java (if not included on the machine), then Jenkins ([here](https://www.jenkins.io/doc/book/installing/linux/)) or as a docker container

    - Start Jenkins using `sudo systemctl start jenkins` as a daemon or a docker container. Now Jenkins UI is avialable on default port 8080. You need to SSH into the machine to obtain the admin password
    - Install plugins: 
        - jdk -> Eclipse Temurin Installer
        - maven -> Config File Provider, Pipeline Maven Integration
        - sonar -> SonarQube Scanner
        - docker -> Docker, Docker Pipeline, docker-build-step
        - kubernetes -> Kubernetes, Kubernetes CLI, Kubernetes Client API, Kubernetes Credentials
    - Configure the plugins
    - Install Trivy on Jenkins server (there is no plugin for this)

    - Configure one Jenkins pipeline per application and save it in a file at the root of that applications:
       - Stages: 
            - Git checkout (provide PAT if repo is private)
            - Build from source code
            - Run tests
            - Run Trivy and output the report in html  
            - Run SonarQube analysis: configure Jenkins to be able to connect to sonar sever so sonar scanner would run on sonar server 
            - Configure Quality Gate: set condition on quality of analysis to pass or fail
            - Package the code 
            - (For Maven) Send artifacts to Nexus: configure Jenkins to do so
            - Build the Docker image with tag and upload it
            - Authenticate Jenkins to deploy the app to K8s
            - Configure E-mail notification. Use Jenkins UI for this
    - (Optional) If you have a domain name, you can install Nginx reverse proxy on Jenkins server so you dont have to use IP address of the server directly. Then you can install `Certbot` that gives you a SSL certificate (using Let's Encrypt) for your domain name and installs it into nginx
    - Create the first Admin User on the Jenkins UI

- (Optional) If you want to keep Jenkins server separate and not run some pipeline task on it (sush as Trivy), you can create a separate virtual machine (Jenkins agent), create non-root user with sudo previlages, install Java, Docker and then connect Jenkins server to agents via SSH.  Now test to see you can run a 'Hello World' pipeline job using Jenkins

- On another virtual machine, set up SonarQube server:
    - Install PostgreSQL and Java
    - Download, extract and start SonarQube. By default, it runs on port 9000. Connect it to PostgreSQL database
    - You can repeat the steps for installing Jenkins: Nginx + Certifcate
    - Confgiure Jenkins to send the application code to Sonar server
        - Generate a token from SonarQube server/security and add the token to Jenkins Credentials
        - From Jenkins Plugins page, install plugins for sonarqube which are SonarQube Scanner, Sonar Quality Gates, Quality Gates 
        - Add credential id + desired shell commands for sonar server to the Jenkins pipeline and sonar server url to Jenkins

- Create a K8s Cluster. On K8s cluster: 
    - Install ArgoCD (if you dont want Jenkins server directly run kubectl commands to deploy apps). You can use a operator controller to install ArgoCD
    - Configure TLS using cert-manager
    - Install Nginx controller as loadbalancer and configure an ingress to expose the app

- On a separate machine, set up Monitoring:
    - Install Grafana, Prometheous, Blackbox Exporter
    - Edit prometheous.yaml to get prometheous scrape blackbox exporter, restart prometheous
    - Connect Grafana to Prometheous to set up dashboards using blackbox. Run node_exporter on other nodes (Jenkins Server etc.) to have them send their info to Prometheous

Note: It is recommende to use IAM Role to grant permissions instad of tokens. But if case of personal project, you can use manual tokens to give Jenkins the permission to deploy an app to K8s:

- Create Service Account, a Role and a RoleBinding to authorize the Jenkins user to deploy in a specific namespace

- Create a k8s secret for this service account in that namespace, obtain the token from the secret and use it to authenticate Jenkins to run commands agaist K8s cluster 

- Configure Jenkins to use this token to apply the deployment.yaml to deploy the app. Also dont forget to install kubectl on Jenkins server


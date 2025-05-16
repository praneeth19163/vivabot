-- MySQL dump 10.13  Distrib 8.0.32, for Win64 (x86_64)
--
-- Host: localhost    Database: vivabot
-- ------------------------------------------------------
-- Server version	8.0.32

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `viva_sessions`
--

DROP TABLE IF EXISTS `viva_sessions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `viva_sessions` (
  `id` int NOT NULL AUTO_INCREMENT,
  `day_of_week` varchar(10) DEFAULT NULL,
  `class_name` varchar(50) DEFAULT NULL,
  `subject` varchar(100) DEFAULT NULL,
  `start_time` time DEFAULT '12:00:00',
  `end_time` time DEFAULT '14:00:00',
  `faculty_name` varchar(100) DEFAULT NULL,
  `faculty_email` varchar(255) NOT NULL DEFAULT 'addulapraneethkumarreddy@gmail.com',
  `end_early` tinyint(1) DEFAULT '0',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=22 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `viva_sessions`
--

LOCK TABLES `viva_sessions` WRITE;
/*!40000 ALTER TABLE `viva_sessions` DISABLE KEYS */;
INSERT INTO `viva_sessions` VALUES (1,'Monday','4CSE-A','ML','09:00:00','15:00:00','E Mahender','addulapraneethkumarreddy@gmail.com',0),(2,'Monday','4CSE-C','BDA_CC','09:00:00','15:00:00','Krishna Rao, Durga Prasad','addulapraneethkumarreddy@gmail.com',0),(3,'Monday','4CSE-D','ML','09:00:00','15:00:00','M Srinivas','addulapraneethkumarreddy@gmail.com',0),(4,'Tuesday','4CSE-A','BDA_CC','13:40:00','16:40:00','Usha Sree, K Srinivas','addulapraneethkumarreddy@gmail.com',0),(5,'Tuesday','4CSE-B','ML','13:40:00','16:40:00','E Mahender','addulapraneethkumarreddy@gmail.com',0),(6,'Wednesday','4CSE-B','BDA_CC','11:00:00','13:00:00','Usha Sree, Santoshi','addulapraneethkumarreddy@gmail.com',0),(7,'Wednesday','4CSE-C','ML','09:00:00','13:00:00','M Srinivas','addulapraneethkumarreddy@gmail.com',0),(8,'Wednesday','4CSE-E','BDA_CC','13:40:00','16:40:00','Chandra Shekar, Anusha','addulapraneethkumarreddy@gmail.com',0),(9,'Thursday','4CSE-D','BDA_CC','13:40:00','16:40:00','Krishna Rao, Durga Prasad','addulapraneethkumarreddy@gmail.com',0),(10,'Friday','4CSE-C','ML','20:00:00','22:00:00','M Srinivas','addulapraneethkumarreddy@gmail.com',1),(11,'Saturday','4CSE-A','BDA_CC','10:00:00','12:00:00','Durga Prasad','addulapraneethkumarreddy@gmail.com',0),(12,'Saturday','4CSE-C','ML','07:00:00','21:00:00','Srinivas','addulapraneethkumarreddy@gmail.com',0),(13,'Saturday','4CSE-D','ML','13:00:00','15:00:00','Sravan Kumar','addulapraneethkumarreddy@gmail.com',0),(14,'Sunday','4CSE-C','ML','07:00:00','21:00:00','Srinivas','addulapraneethkumarreddy@gmail.com',0),(15,'Sunday','4CSE-C','BDA_CC','07:00:00','21:00:00','Krishna Rao, Durga Prasad','addulapraneethkumarreddy@gmail.com',0),(16,'Sunday','4CSE-E','ML','19:00:00','21:00:00','Mahender','addulapraneethkumarreddy@gmail.com',0),(17,'Monday','4CSE-C','ML','09:00:00','15:00:00','Srinivas','addulapraneethkumarreddy@gmail.com',0),(18,'Tuesday','4CSE-C','ML','08:00:00','21:00:00','Srinivas','addulapraneethkumarreddy@gmail.com',0),(19,'Saturday','4CSE-E','ML','09:00:00','15:00:00','Mahender','addulapraneethkumarreddy@gmail.com',0),(20,'Thursday','4CSE-C','ML','09:00:00','15:00:00','Srinivas','addulapraneethkumarreddy@gmail.com',0),(21,'Monday','4CSE-C','RIDDLE','09:00:00','21:00:00','Lohitha','addulapraneethkumarreddy@gmail.com',0);
/*!40000 ALTER TABLE `viva_sessions` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-04-25 22:16:43

-- MySQL dump 10.13  Distrib 8.0.32, for Win64 (x86_64)
--
-- Host: localhost    Database: vivabot
-- ------------------------------------------------------
-- Server version	8.0.32

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ml_questions`
--

DROP TABLE IF EXISTS `ml_questions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ml_questions` (
  `qn_no` int NOT NULL AUTO_INCREMENT,
  `week` varchar(10) NOT NULL,
  `question` text NOT NULL,
  `proficiency_level` enum('Easy','Medium','Hard') NOT NULL,
  PRIMARY KEY (`qn_no`)
) ENGINE=InnoDB AUTO_INCREMENT=45 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `ml_questions`
--

LOCK TABLES `ml_questions` WRITE;
/*!40000 ALTER TABLE `ml_questions` DISABLE KEYS */;
INSERT INTO `ml_questions` VALUES (1,'Week 1','What is Naive Bayes?','Easy'),(2,'Week 1','What are the assumptions of Naive Bayes?','Easy'),(3,'Week 1','What is prior probability in Naive Bayes?','Easy'),(4,'Week 1','What is conditional probability in Naive Bayes?','Easy'),(5,'Week 1','Why is Naive Bayes called naive?','Easy'),(6,'Week 1','What are the advantages of using Naive Bayes?','Medium'),(7,'Week 1','What is the Bayes theorem used in Naive Bayes?','Medium'),(8,'Week 1','What are the different types of Naive Bayes classifiers?','Medium'),(9,'Week 1','How does Laplace smoothing help in Naive Bayes?','Medium'),(10,'Week 1','How does Naive Bayes handle missing data?','Medium'),(11,'Week 1','Explain the steps involved in implementing the Naive Bayes classifier.','Hard'),(12,'Week 1','How does Naive Bayes perform when features are correlated?','Hard'),(13,'Week 1','How do you calculate posterior probability in Naive Bayes?','Hard'),(14,'Week 1','What are the limitations of the Naive Bayes algorithm?','Hard'),(15,'Week 1','How can Naive Bayes be improved for text classification?','Hard'),(16,'Week 2','What is a decision tree?','Easy'),(17,'Week 2','What are the key components of a decision tree?','Easy'),(18,'Week 2','What is entropy in a decision tree?','Easy'),(19,'Week 2','What is information gain in a decision tree?','Easy'),(20,'Week 2','How do decision trees handle categorical and numerical data?','Easy'),(21,'Week 2','What is the difference between Gini index and entropy?','Medium'),(22,'Week 2','What are the advantages of decision trees?','Medium'),(23,'Week 2','What are the disadvantages of decision trees?','Medium'),(24,'Week 2','How does overfitting occur in decision trees?','Medium'),(25,'Week 2','Explain the steps in building a decision tree.','Hard'),(26,'Week 2','How is the best attribute selected in a decision tree?','Hard'),(27,'Week 2','How do decision trees handle missing values?','Hard'),(28,'Week 2','How do you avoid overfitting in decision trees?','Hard'),(29,'Week 2','What is a random forest, and how is it related to decision trees?','Hard'),(30,'Week 3','Can you explain what a Naive Bayes Classifier is and give an example of its use?','Easy'),(31,'Week 3','What is the key assumption made by a Naive Bayes Classifier?','Easy'),(32,'Week 3','What is the formula for calculating the probability of a class using a Naive Bayes Classifier?','Easy'),(33,'Week 3','How does a Multinomial Naive Bayes Classifier differ from a Bernoulli Naive Bayes Classifier?','Easy'),(34,'Week 3','What is smoothing and why is it important in a Naive Bayes Classifier?','Easy'),(35,'Week 3','Why is the Naive Bayes Classifier less susceptible to overfitting compared to other machine learning algorithms?','Medium'),(36,'Week 3','In a Naive Bayes Classifier, how do we handle categorical features with more than two categories?','Medium'),(37,'Week 3','What are the limitations of the Naive Bayes Classifier and when might it be inappropriate to use one?','Medium'),(38,'Week 3','Can you describe a real-world scenario where using a Naive Bayes Classifier would be beneficial and explain why?','Medium'),(39,'Week 3','How can we perform cross-validation on a Naive Bayes Classifier, and what is the purpose of doing so?','Medium'),(40,'Week 3','What are some extensions of the basic Naive Bayes Classifier that have been developed to improve its performance?','Hard'),(41,'Week 3','In the context of text classification, how does the Complement Naive Bayes algorithm work and when is it useful?','Hard'),(42,'Week 3','Explain the concept of the Gaussian Naive Bayes (GNB) and provide an example where it would be appropriate to use GNB over a standard Naive Bayes Classifier.','Hard'),(43,'Week 3','What is the issue with using the Laplacian smoothing method in text classification, and why might it lead to poor results?','Hard'),(44,'Week 3','Can you explain the concept of Multiple Instance Learning (MIL) and how it relates to Naive Bayes Classifiers?','Hard');
/*!40000 ALTER TABLE `ml_questions` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-05-16 14:23:40

-- MySQL dump 10.13  Distrib 8.0.32, for Win64 (x86_64)
--
-- Host: localhost    Database: vivabot
-- ------------------------------------------------------
-- Server version	8.0.32

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `bda_cc_questions`
--

DROP TABLE IF EXISTS `bda_cc_questions`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `bda_cc_questions` (
  `qn_no` int NOT NULL AUTO_INCREMENT,
  `week` varchar(10) DEFAULT NULL,
  `question` text,
  `proficiency_level` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`qn_no`)
) ENGINE=InnoDB AUTO_INCREMENT=46 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `bda_cc_questions`
--

LOCK TABLES `bda_cc_questions` WRITE;
/*!40000 ALTER TABLE `bda_cc_questions` DISABLE KEYS */;
INSERT INTO `bda_cc_questions` VALUES (1,'Week 1','BDA: Installation, Configuration, and Running of Hadoop and HDFS. CC: Create Virtual machines using Open source software: VM Ware/ Oracle Virtual Box.','Easy'),(2,'Week 1','BDA: Set up Hadoop environment and configure HDFS. CC: Use Amazon EC2 to create a Virtual machine and configure basic network settings.','Easy'),(3,'Week 1','BDA: Optimize Hadoop and HDFS configurations for scalability. CC: Configure security settings for EC2 virtual machines.','Easy'),(4,'Week 1','BDA: Install and configure Hadoop and HDFS for large datasets. CC: Set up Amazon EC2 for running virtual machines with Linux environments.','Medium'),(5,'Week 1','BDA: Configure Hadoop clusters for high availability and fault tolerance. CC: Use EC2 for cloud-based virtual machine management.','Medium'),(6,'Week 1','BDA: Optimize Hadoop performance for distributed data storage. CC: Automate EC2 virtual machine deployment using cloud templates.','Medium'),(7,'Week 1','BDA: Advanced optimization of Hadoop and HDFS for real-time data processing. CC: Secure EC2 instances and optimize their network configuration for performance.','Hard'),(8,'Week 1','BDA: Implement Hadoop performance tuning for distributed data processing systems. CC: Automate multi-VM network setup with EC2 instances for large-scale operations.','Hard'),(9,'Week 1','BDA: Implement Hadoop on top of a large data set with fault tolerance and performance monitoring. CC: Secure EC2 instances with cloud security tools for advanced configurations.','Hard'),(10,'Week 2','BDA: Implement file management tasks in Hadoop: Adding files and directories. CC: Use Amazon EC2 to create and manage Virtual machines.','Easy'),(11,'Week 2','BDA: Add directories and manage file permissions in HDFS. CC: Use EC2 to create and configure Linux-based virtual machines for cloud operations.','Easy'),(12,'Week 2','BDA: Implement Hadoop file management techniques. CC: Use EC2 to set up basic networking between virtual machines for data storage.','Easy'),(13,'Week 2','BDA: Implement file retrieval and deletion in Hadoop using HDFS commands. CC: Integrate EC2 virtual machines for cloud-based data storage management.','Medium'),(14,'Week 2','BDA: Troubleshoot Hadoop file system issues using HDFS commands. CC: Implement security features for EC2 instances and S3 bucket access.','Medium'),(15,'Week 2','BDA: Implement a scalable data pipeline for file management in Hadoop. CC: Automate EC2 virtual machine deployment and management.','Medium'),(16,'Week 2','BDA: Implement Hadoop file management for real-time data processing tasks. CC: Secure EC2 instances with encryption and firewalls for data integrity.','Hard'),(17,'Week 2','BDA: Build scalable, fault-tolerant file management systems in Hadoop. CC: Implement cloud-based security protocols for EC2 and storage systems.','Hard'),(18,'Week 2','BDA: Optimize file retrieval and deletion processes in Hadoop for large datasets. CC: Set up EC2 and S3 integration for automated cloud storage operations.','Hard'),(19,'Week 3','BDA: Implementation of Word Count / Frequency Programs using MapReduce. CC: Use Amazon S3 to create buckets and upload objects.','Easy'),(20,'Week 3','BDA: Write a simple word count program using MapReduce in Hadoop. CC: Set up Amazon S3 buckets for storing word count data.','Easy'),(21,'Week 3','BDA: Implement a frequency count program for large datasets in Hadoop. CC: Upload and retrieve data from S3 for word count processing.','Easy'),(22,'Week 3','BDA: Design a MapReduce program to process and count word frequencies in large data sets. CC: Integrate S3 buckets for storage and retrieval of large word count data.','Medium'),(23,'Week 3','BDA: Implement word frequency analysis using MapReduce for a large-scale dataset. CC: Configure advanced features in S3 for data storage and processing.','Medium'),(24,'Week 3','BDA: Optimize a MapReduce program for distributed word frequency counting. CC: Automate S3 bucket management for large data ingestion and retrieval.','Medium'),(25,'Week 3','BDA: Implement a scalable MapReduce solution for real-time word count applications. CC: Set up S3 buckets with high availability and low-latency configurations.','Hard'),(26,'Week 3','BDA: Optimize MapReduce performance for processing large-scale word frequency data. CC: Use advanced S3 storage strategies for large data processing tasks.','Hard'),(27,'Week 3','BDA: Build a fault-tolerant MapReduce program for word frequency processing on Hadoop. CC: Implement S3 bucket versioning and replication for data backup.','Hard'),(28,'Week 4','BDA: Implement MR Program that processes a Weather Dataset. CC: Install the Simple Notification Service on Ubuntu.','Easy'),(29,'Week 4','BDA: Write a MapReduce program to process weather data in Hadoop. CC: Set up Simple Notification Service on Ubuntu to handle notifications.','Easy'),(30,'Week 4','BDA: Implement a basic MR program to analyze weather patterns. CC: Configure SNS on Ubuntu for basic notification functionality.','Easy'),(31,'Week 4','BDA: Implement a real-time MapReduce program to process weather data. CC: Use SNS to send notifications based on weather conditions.','Medium'),(32,'Week 4','BDA: Process large weather datasets in Hadoop with MapReduce and optimize performance. CC: Set up SNS for scalable notification services in cloud applications.','Medium'),(33,'Week 4','BDA: Use MapReduce to analyze weather data from multiple sources. CC: Automate SNS deployment and management for real-time notifications on Ubuntu.','Medium'),(34,'Week 4','BDA: Build a large-scale weather data processing system with MapReduce. CC: Secure SNS deployment and implement disaster recovery solutions for Ubuntu systems.','Hard'),(35,'Week 4','BDA: Optimize weather data processing using MapReduce and integrate it with other big data tools. CC: Use advanced SNS configurations for enterprise-level messaging systems.','Hard'),(36,'Week 4','BDA: Implement a highly available weather data processing system with Hadoop MapReduce. CC: Set up SNS with multi-region support for notifications in cloud environments.','Hard'),(37,'Week 5','BDA: Install and Run Pig then write Pig Latin scripts to sort, group, join, project, and filter your data. CC: Use Amazon CloudFront to create a Distribution and Use Amazon Route53 to create a domain.','Easy'),(38,'Week 5','BDA: Install Apache Pig and write basic Pig Latin scripts to handle data. CC: Set up Amazon CloudFront for distributing content and Route53 for DNS management.','Easy'),(39,'Week 5','BDA: Write Pig Latin scripts to sort and filter data in Hadoop. CC: Use Route53 to create a simple domain and CloudFront for CDN configuration.','Easy'),(40,'Week 5','BDA: Implement Pig Latin scripts for advanced data manipulation. CC: Set up CloudFront and Route53 to handle large-scale content delivery and domain management.','Medium'),(41,'Week 5','BDA: Optimize Pig Latin scripts for handling complex data transformations. CC: Use Route53 for managing custom domain configurations and integrate it with CloudFront.','Medium'),(42,'Week 5','BDA: Implement scalable data filtering and grouping using Pig in Hadoop. CC: Automate CloudFront and Route53 deployment for better cloud resource management.','Medium'),(43,'Week 5','BDA: Write complex Pig Latin scripts for real-time data analysis and reporting. CC: Set up CloudFront with advanced security features and use Route53 for efficient DNS routing.','Hard'),(44,'Week 5','BDA: Optimize Pig Latin scripts for large data pipelines and complex workflows. CC: Implement CloudFront distributions with global routing and secure domain management using Route53.','Hard'),(45,'Week 5','BDA: Build advanced Pig processing workflows for enterprise data integration. CC: Use CloudFront for enterprise-level content delivery and Route53 for reliable domain management.','Hard');
/*!40000 ALTER TABLE `bda_cc_questions` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-05-16 14:23:40


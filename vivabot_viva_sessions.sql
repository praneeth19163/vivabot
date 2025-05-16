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

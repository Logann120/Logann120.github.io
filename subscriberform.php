<?php
// The message
$message = $_POST['message'];

// In case any of our lines are larger than 70 characters, we should use wordwrap()
$message = wordwrap($message, 70, "\r\n");

// Send
mail('logannoonan120@gmail.com', 'My Subject', $message);
?>

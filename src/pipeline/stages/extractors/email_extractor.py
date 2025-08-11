"""
Enhanced email content extractor with comprehensive parsing
"""

import re
import html
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime, parseaddr

from ...core.models import DocumentContent, ExtractedContent
from ...core.interfaces import DocumentType
from ...core.exceptions import ContentExtractionError
from ...core.logging_utils import get_pipeline_logger
from ...core.utils import calculate_content_hash, timing_decorator


@dataclass
class EmailHeader:
    """Enhanced email header information"""
    subject: str
    from_addr: str
    from_name: str
    to_addrs: List[str]
    cc_addrs: List[str]
    bcc_addrs: List[str]
    reply_to: str
    date: Optional[datetime]
    message_id: str
    in_reply_to: str
    references: List[str]
    priority: str
    content_type: str
    encoding: str


@dataclass
class EmailAttachment:
    """Email attachment information"""
    filename: str
    content_type: str
    size_bytes: int
    content_id: Optional[str] = None
    is_inline: bool = False


@dataclass
class EmailStructure:
    """Email structure information"""
    is_multipart: bool
    parts_count: int
    has_html: bool
    has_plain_text: bool
    has_attachments: bool
    attachments: List[EmailAttachment]
    total_size: int
    thread_info: Dict[str, Any]


class EnhancedEmailExtractor:
    """Enhanced email content extractor with comprehensive parsing"""
    
    def __init__(self):
        self.logger = get_pipeline_logger()
    
    @timing_decorator
    def extract_content(self, document: DocumentContent) -> ExtractedContent:
        """Extract content from email with enhanced parsing"""
        
        self.logger.info("Starting enhanced email content extraction")
        
        try:
            import email
            from email import policy
            
            # Parse email content with modern policy
            msg = email.message_from_bytes(document.raw_content, policy=policy.default)
            
            # Extract enhanced headers
            headers = self._extract_enhanced_headers(msg)
            
            # Analyze email structure
            structure = self._analyze_email_structure(msg)
            
            # Extract body content with different formats
            body_content = self._extract_body_content(msg)
            
            # Extract attachments information
            attachments = self._extract_attachments_info(msg)
            
            # Build sections
            sections = self._build_sections(headers, body_content, attachments)
            
            # Combine all text content
            text_content = self._build_text_content(headers, body_content)
            
            # Enhanced metadata
            metadata = {
                'headers': {
                    'subject': headers.subject,
                    'from': headers.from_addr,
                    'from_name': headers.from_name,
                    'to': headers.to_addrs,
                    'cc': headers.cc_addrs,
                    'bcc': headers.bcc_addrs,
                    'reply_to': headers.reply_to,
                    'date': headers.date.isoformat() if headers.date else None,
                    'message_id': headers.message_id,
                    'in_reply_to': headers.in_reply_to,
                    'references': headers.references,
                    'priority': headers.priority,
                    'content_type': headers.content_type,
                    'encoding': headers.encoding
                },
                'structure': {
                    'is_multipart': structure.is_multipart,
                    'parts_count': structure.parts_count,
                    'has_html': structure.has_html,
                    'has_plain_text': structure.has_plain_text,
                    'has_attachments': structure.has_attachments,
                    'total_size': structure.total_size,
                    'thread_info': structure.thread_info
                },
                'attachments': [
                    {
                        'filename': att.filename,
                        'content_type': att.content_type,
                        'size_bytes': att.size_bytes,
                        'is_inline': att.is_inline
                    }
                    for att in attachments
                ],
                'content_analysis': {
                    'word_count': len(text_content.split()) if text_content else 0,
                    'char_count': len(text_content) if text_content else 0,
                    'has_urls': self._has_urls(text_content),
                    'has_phone_numbers': self._has_phone_numbers(text_content),
                    'has_email_addresses': self._has_email_addresses(text_content),
                    'language_detected': self._detect_language(text_content),
                    'sentiment_indicators': self._extract_sentiment_indicators(text_content)
                },
                'extraction_method': 'Enhanced email.parser',
                'security_analysis': {
                    'spf_pass': self._check_spf_status(msg),
                    'dkim_valid': self._check_dkim_status(msg),
                    'suspicious_links': self._detect_suspicious_links(text_content),
                    'potential_phishing': self._detect_phishing_indicators(headers, text_content)
                }
            }
            
            return ExtractedContent(
                document_id=calculate_content_hash(document.raw_content),
                document_type=DocumentType.EMAIL.value,
                text_content=text_content,
                pages=None,  # Emails don't have pages
                sections=sections,
                metadata=metadata
            )
        
        except Exception as e:
            raise ContentExtractionError(f"Enhanced email extraction failed: {e}", DocumentType.EMAIL.value)
    
    def _extract_enhanced_headers(self, msg) -> EmailHeader:
        """Extract enhanced header information"""
        
        # Parse from address
        from_name, from_addr = parseaddr(msg.get('From', ''))
        
        # Parse to addresses
        to_addrs = [addr.strip() for addr in msg.get('To', '').split(',') if addr.strip()]
        cc_addrs = [addr.strip() for addr in msg.get('CC', '').split(',') if addr.strip()]
        bcc_addrs = [addr.strip() for addr in msg.get('BCC', '').split(',') if addr.strip()]
        
        # Parse date
        date_str = msg.get('Date', '')
        date_obj = None
        if date_str:
            try:
                date_obj = parsedate_to_datetime(date_str)
            except:
                pass
        
        # Parse references
        references = []
        refs_str = msg.get('References', '')
        if refs_str:
            references = [ref.strip('<>') for ref in refs_str.split() if ref.strip()]
        
        return EmailHeader(
            subject=msg.get('Subject', ''),
            from_addr=from_addr,
            from_name=from_name,
            to_addrs=to_addrs,
            cc_addrs=cc_addrs,
            bcc_addrs=bcc_addrs,
            reply_to=msg.get('Reply-To', ''),
            date=date_obj,
            message_id=msg.get('Message-ID', ''),
            in_reply_to=msg.get('In-Reply-To', ''),
            references=references,
            priority=msg.get('X-Priority', msg.get('Priority', '')),
            content_type=msg.get_content_type(),
            encoding=msg.get_charset() or 'utf-8'
        )
    
    def _analyze_email_structure(self, msg) -> EmailStructure:
        """Analyze email structure"""
        
        is_multipart = msg.is_multipart()
        parts_count = 0
        has_html = False
        has_plain_text = False
        has_attachments = False
        attachments = []
        total_size = 0
        
        if is_multipart:
            for part in msg.walk():
                parts_count += 1
                content_type = part.get_content_type()
                
                if content_type == 'text/plain':
                    has_plain_text = True
                elif content_type == 'text/html':
                    has_html = True
                elif part.get_filename():
                    has_attachments = True
                    attachments.append(self._extract_attachment_info(part))
                
                # Estimate size
                try:
                    content = part.get_content()
                    if isinstance(content, str):
                        total_size += len(content.encode('utf-8'))
                    elif isinstance(content, bytes):
                        total_size += len(content)
                except:
                    pass
        else:
            parts_count = 1
            content_type = msg.get_content_type()
            if content_type == 'text/plain':
                has_plain_text = True
            elif content_type == 'text/html':
                has_html = True
        
        # Analyze thread information
        thread_info = self._analyze_thread_info(msg)
        
        return EmailStructure(
            is_multipart=is_multipart,
            parts_count=parts_count,
            has_html=has_html,
            has_plain_text=has_plain_text,
            has_attachments=has_attachments,
            attachments=attachments,
            total_size=total_size,
            thread_info=thread_info
        )
    
    def _extract_body_content(self, msg) -> Dict[str, str]:
        """Extract body content in different formats"""
        body_content = {
            'plain_text': '',
            'html': '',
            'combined': ''
        }
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                
                if content_type == 'text/plain':
                    try:
                        content = part.get_content()
                        if content:
                            body_content['plain_text'] += content + '\n\n'
                    except:
                        pass
                
                elif content_type == 'text/html':
                    try:
                        content = part.get_content()
                        if content:
                            body_content['html'] += content + '\n\n'
                            # Convert HTML to plain text for combined view
                            plain_from_html = self._html_to_text(content)
                            body_content['combined'] += plain_from_html + '\n\n'
                    except:
                        pass
        else:
            try:
                content = msg.get_content()
                content_type = msg.get_content_type()
                
                if content_type == 'text/plain':
                    body_content['plain_text'] = content
                    body_content['combined'] = content
                elif content_type == 'text/html':
                    body_content['html'] = content
                    body_content['combined'] = self._html_to_text(content)
            except:
                pass
        
        # Clean up content
        for key in body_content:
            body_content[key] = body_content[key].strip()
        
        return body_content
    
    def _extract_attachments_info(self, msg) -> List[EmailAttachment]:
        """Extract attachment information"""
        attachments = []
        
        if msg.is_multipart():
            for part in msg.walk():
                filename = part.get_filename()
                if filename:
                    attachment = self._extract_attachment_info(part)
                    attachments.append(attachment)
        
        return attachments
    
    def _extract_attachment_info(self, part) -> EmailAttachment:
        """Extract information about a single attachment"""
        filename = part.get_filename() or 'unknown'
        content_type = part.get_content_type()
        content_id = part.get('Content-ID', '').strip('<>')
        
        # Estimate size
        size_bytes = 0
        try:
            content = part.get_content()
            if isinstance(content, bytes):
                size_bytes = len(content)
            elif isinstance(content, str):
                size_bytes = len(content.encode('utf-8'))
        except:
            pass
        
        # Check if inline
        disposition = part.get('Content-Disposition', '')
        is_inline = 'inline' in disposition.lower()
        
        return EmailAttachment(
            filename=filename,
            content_type=content_type,
            size_bytes=size_bytes,
            content_id=content_id,
            is_inline=is_inline
        )
    
    def _build_sections(self, headers: EmailHeader, body_content: Dict[str, str], attachments: List[EmailAttachment]) -> Dict[str, str]:
        """Build sections dictionary"""
        sections = {}
        
        # Header section
        header_text = f"Subject: {headers.subject}\n"
        header_text += f"From: {headers.from_name} <{headers.from_addr}>\n"
        header_text += f"To: {', '.join(headers.to_addrs)}\n"
        if headers.cc_addrs:
            header_text += f"CC: {', '.join(headers.cc_addrs)}\n"
        if headers.date:
            header_text += f"Date: {headers.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        sections['headers'] = header_text
        
        # Body sections
        if body_content['plain_text']:
            sections['body_plain'] = body_content['plain_text']
        
        if body_content['html']:
            sections['body_html'] = body_content['html']
        
        if body_content['combined']:
            sections['body'] = body_content['combined']
        
        # Attachments section
        if attachments:
            att_text = f"Attachments ({len(attachments)}):\n"
            for att in attachments:
                att_text += f"- {att.filename} ({att.content_type}, {att.size_bytes} bytes)\n"
            sections['attachments'] = att_text
        
        return sections
    
    def _build_text_content(self, headers: EmailHeader, body_content: Dict[str, str]) -> str:
        """Build combined text content"""
        parts = []
        
        # Add header information
        parts.append(f"Subject: {headers.subject}")
        parts.append(f"From: {headers.from_name} <{headers.from_addr}>")
        parts.append(f"To: {', '.join(headers.to_addrs)}")
        if headers.date:
            parts.append(f"Date: {headers.date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        parts.append("")  # Empty line
        
        # Add body content (prefer plain text, fall back to HTML converted to text)
        if body_content['plain_text']:
            parts.append(body_content['plain_text'])
        elif body_content['combined']:
            parts.append(body_content['combined'])
        
        return '\n'.join(parts)
    
    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML content to plain text"""
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', html_content)
            # Decode HTML entities
            text = html.unescape(text)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except:
            return html_content
    
    def _analyze_thread_info(self, msg) -> Dict[str, Any]:
        """Analyze email thread information"""
        thread_info = {
            'is_reply': bool(msg.get('In-Reply-To')),
            'is_forward': 'fwd:' in msg.get('Subject', '').lower() or 'fw:' in msg.get('Subject', '').lower(),
            'thread_depth': len(msg.get('References', '').split()) if msg.get('References') else 0,
            'conversation_id': msg.get('Thread-Index', ''),
        }
        
        return thread_info
    
    def _has_urls(self, text: str) -> bool:
        """Check if text contains URLs"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return bool(re.search(url_pattern, text, re.IGNORECASE))
    
    def _has_phone_numbers(self, text: str) -> bool:
        """Check if text contains phone numbers"""
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
            r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',  # (123) 456-7890
            r'\b\+\d{1,3}[-.\s]?\d{1,14}\b'  # International
        ]
        
        for pattern in phone_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def _has_email_addresses(self, text: str) -> bool:
        """Check if text contains email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return bool(re.search(email_pattern, text))
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words"""
        if not text:
            return 'unknown'
        
        # Simple heuristic based on common words
        english_words = ['the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with']
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no']
        french_words = ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir']
        
        text_lower = text.lower()
        
        english_count = sum(1 for word in english_words if word in text_lower)
        spanish_count = sum(1 for word in spanish_words if word in text_lower)
        french_count = sum(1 for word in french_words if word in text_lower)
        
        if english_count >= spanish_count and english_count >= french_count:
            return 'english'
        elif spanish_count >= french_count:
            return 'spanish'
        elif french_count > 0:
            return 'french'
        else:
            return 'unknown'
    
    def _extract_sentiment_indicators(self, text: str) -> Dict[str, int]:
        """Extract basic sentiment indicators"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'pleased']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry', 'frustrated', 'disappointed', 'upset']
        urgent_words = ['urgent', 'asap', 'immediately', 'emergency', 'critical', 'important', 'rush', 'deadline']
        
        text_lower = text.lower()
        
        return {
            'positive_indicators': sum(1 for word in positive_words if word in text_lower),
            'negative_indicators': sum(1 for word in negative_words if word in text_lower),
            'urgent_indicators': sum(1 for word in urgent_words if word in text_lower)
        }
    
    def _check_spf_status(self, msg) -> bool:
        """Check SPF status from headers"""
        received_spf = msg.get('Received-SPF', '').lower()
        return 'pass' in received_spf
    
    def _check_dkim_status(self, msg) -> bool:
        """Check DKIM status from headers"""
        dkim_signature = msg.get('DKIM-Signature', '')
        return bool(dkim_signature)
    
    def _detect_suspicious_links(self, text: str) -> List[str]:
        """Detect potentially suspicious links"""
        suspicious_links = []
        
        # Find all URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        
        suspicious_domains = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co']
        suspicious_patterns = [r'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+']  # IP addresses
        
        for url in urls:
            # Check for suspicious domains
            for domain in suspicious_domains:
                if domain in url.lower():
                    suspicious_links.append(url)
                    break
            
            # Check for suspicious patterns
            for pattern in suspicious_patterns:
                if re.search(pattern, url):
                    suspicious_links.append(url)
                    break
        
        return suspicious_links
    
    def _detect_phishing_indicators(self, headers: EmailHeader, text: str) -> List[str]:
        """Detect potential phishing indicators"""
        indicators = []
        
        # Check for suspicious subject patterns
        subject_lower = headers.subject.lower()
        phishing_subjects = ['urgent action required', 'verify your account', 'suspended account', 'click here now']
        
        for pattern in phishing_subjects:
            if pattern in subject_lower:
                indicators.append(f"Suspicious subject: {pattern}")
        
        # Check for mismatched sender
        if headers.from_name and headers.from_addr:
            if 'paypal' in headers.from_name.lower() and 'paypal.com' not in headers.from_addr.lower():
                indicators.append("Sender name/address mismatch")
        
        # Check for urgent language in body
        urgent_phrases = ['act now', 'limited time', 'expires today', 'immediate action']
        text_lower = text.lower()
        
        for phrase in urgent_phrases:
            if phrase in text_lower:
                indicators.append(f"Urgent language: {phrase}")
        
        return indicators